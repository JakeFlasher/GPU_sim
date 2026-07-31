"""Closed v1 preregistration receipt for target-free evaluator inputs."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    sha256_bytes,
)

from ._common import (
    AuthorityBlockerCode,
    authority_unavailable,
    evaluation_boundary_model,
    require_exact_evaluation_boundary,
)
from ._receipt_common import (
    ArtifactIdentity,
    EvaluatorModelIdentity,
    ReceiptPhase,
    ReceiptValidationSchemas,
    RetainedSchemaInput,
    derive_domain_sha256,
    phase_chain_model,
    require_exact_artifact_kind,
    require_exact_keys,
    require_exact_phase_chain,
    require_ascii_token,
    require_sha256,
)
from .model_seal import (
    seal_evaluator_model,
    strict_json_object,
    verify_evaluator_model,
    verify_retained_schema,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "EVALUATOR_PREREGISTRATION_FIXTURE"
SCHEMA_ID = (
    "https://gpu-simulator.invalid/eval/schemas/"
    "preregistration_receipt.v1.schema.json"
)
SCHEMA_SHA256 = "3cdf0df9500ed1c5704db1c8c125d25524e4bd4712a075577576c08d0e59f5ab"
METHOD_ARTIFACT_KIND = "FIXTURE_UNVALIDATED_EVALUATION_METHOD"
SOURCE_ARTIFACT_KIND = "FIXTURE_UNVALIDATED_SOURCE_INVENTORY"
ANCESTRY_AUDIT_ARTIFACT_KIND = "FIXTURE_UNVALIDATED_ANCESTRY_AUDIT"
FIXTURE_CONTRACT_MODE = "FIXTURE_ONLY"
# The checked-in M5 ancestry schema proves only mechanical non-revival and
# explicitly requires semantic review. No target-free production authority is
# therefore installed here.
PRODUCTION_TARGET_FREE_ARTIFACT_AUTHORITIES = MappingProxyType({})
PRODUCTION_ANCESTRY_REVIEW_AUTHORITIES = MappingProxyType({})

_AUTHORITY_SLOTS = [
    {"authority": "REVIEWER_ACCEPTANCE", "status": "UNRESOLVED"},
    {"authority": "OPERATOR_APPROVAL", "status": "UNRESOLVED"},
    {"authority": "TRUST_ACCEPTANCE", "status": "UNRESOLVED"},
]


def unresolved_authority_slots_model() -> list[dict[str, str]]:
    """Return the exact non-accepting authority state for preregistration."""

    return [dict(slot) for slot in _AUTHORITY_SLOTS]


def build_preregistration_receipt(**_inputs: Any) -> dict[str, Any]:
    raise authority_unavailable(
        AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE,
        "no supported target-free method/source/ancestry authority set exists",
    )


def build_preregistration_receipt_fixture(
    *,
    target_free_method_artifact: ArtifactIdentity,
    source_inventory_artifact: ArtifactIdentity,
    ancestry_audit_artifact: ArtifactIdentity,
    fixture_candidate_inventory: dict[str, Any],
    schemas: ReceiptValidationSchemas,
) -> dict[str, Any]:
    """Pure fixture builder; it performs no reads, approvals, or registration."""

    _verify_schema_input(schemas)
    artifacts = _unvalidated_artifact_fixtures_model(
        target_free_method_artifact=target_free_method_artifact,
        source_inventory_artifact=source_inventory_artifact,
        ancestry_audit_artifact=ancestry_audit_artifact,
    )
    candidate_inventory = _validate_fixture_candidate_inventory(
        fixture_candidate_inventory
    )
    if source_inventory_artifact.artifact_sha256 != sha256_bytes(
        canonical_json_bytes(candidate_inventory)
    ):
        raise ImmutableArtifactError(
            "fixture source inventory digest does not bind candidate inventory"
        )
    authority_slots = unresolved_authority_slots_model()
    chain_id = _derive_chain_id(artifacts, candidate_inventory, authority_slots)
    return seal_evaluator_model(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "schema_sha256": SCHEMA_SHA256,
            "contract_mode": FIXTURE_CONTRACT_MODE,
            "production_eligibility": {
                "status": "UNAVAILABLE",
                "reason_codes": [
                    "TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE",
                    "ANCESTRY_REVIEW_AUTHORITY_UNAVAILABLE",
                ],
            },
            "evaluation_boundary": evaluation_boundary_model(),
            "phase_chain": phase_chain_model(
                phase=ReceiptPhase.PREREGISTRATION,
                chain_id_sha256=chain_id,
                predecessor=None,
            ),
            "unvalidated_artifact_fixtures": artifacts,
            "fixture_candidate_inventory": candidate_inventory,
            "authority_slots": authority_slots,
        }
    )


def verify_preregistration_receipt_model(
    model: dict[str, Any], *, schemas: ReceiptValidationSchemas
) -> None:
    del model, schemas
    raise authority_unavailable(
        AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE,
        "target-free method/source/ancestry authorities are not installed",
    )


def verify_preregistration_receipt_fixture_model(
    model: dict[str, Any], *, schemas: ReceiptValidationSchemas
) -> None:
    schema = _verify_schema_input(schemas)
    verify_evaluator_model(
        model,
        expected_schema_version=SCHEMA_VERSION,
        expected_artifact_kind=ARTIFACT_KIND,
        expected_schema_sha256=SCHEMA_SHA256,
        expected_schema_id=SCHEMA_ID,
        schema_bytes=schema.schema_bytes,
        schema_snapshot=schema.schema_snapshot,
    )
    require_exact_keys(
        model,
        {
            "schema_version",
            "artifact_kind",
            "schema_sha256",
            "contract_mode",
            "production_eligibility",
            "evaluation_boundary",
            "phase_chain",
            "unvalidated_artifact_fixtures",
            "fixture_candidate_inventory",
            "authority_slots",
            "model_seal_sha256",
        },
        "preregistration receipt",
    )
    if model["contract_mode"] != FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("preregistration receipt is not fixture-only")
    if model["production_eligibility"] != {
        "status": "UNAVAILABLE",
        "reason_codes": [
            "TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE",
            "ANCESTRY_REVIEW_AUTHORITY_UNAVAILABLE",
        ],
    }:
        raise ImmutableArtifactError("preregistration production blocker mismatch")
    require_exact_evaluation_boundary(model["evaluation_boundary"])
    artifacts = _parse_unvalidated_artifact_fixtures(
        model["unvalidated_artifact_fixtures"]
    )
    candidate_inventory = _validate_fixture_candidate_inventory(
        model["fixture_candidate_inventory"]
    )
    source_identity = ArtifactIdentity.from_model(
        artifacts["source_inventory"], label="fixture source inventory identity"
    )
    if source_identity.artifact_sha256 != sha256_bytes(
        canonical_json_bytes(candidate_inventory)
    ):
        raise ImmutableArtifactError(
            "fixture source inventory digest does not bind candidate inventory"
        )
    if model["authority_slots"] != unresolved_authority_slots_model():
        raise ImmutableArtifactError("preregistration authority slots are not unresolved")
    chain_id = _derive_chain_id(
        artifacts, candidate_inventory, model["authority_slots"]
    )
    require_exact_phase_chain(
        model["phase_chain"],
        phase=ReceiptPhase.PREREGISTRATION,
        chain_id_sha256=chain_id,
        predecessor=None,
    )


def load_preregistration_receipt(
    payload: bytes, *, schemas: ReceiptValidationSchemas
) -> dict[str, Any]:
    del payload, schemas
    raise authority_unavailable(
        AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE,
        "production preregistration loading is blocked by missing authorities",
    )


def load_preregistration_receipt_fixture(
    payload: bytes, *, schemas: ReceiptValidationSchemas
) -> dict[str, Any]:
    model = strict_json_object(payload, label="evaluator preregistration fixture")
    verify_preregistration_receipt_fixture_model(model, schemas=schemas)
    return model


def preregistration_fixture_identity(
    model: dict[str, Any], *, schemas: ReceiptValidationSchemas
) -> EvaluatorModelIdentity:
    verify_preregistration_receipt_fixture_model(model, schemas=schemas)
    return EvaluatorModelIdentity.create(
        model,
        expected_schema_version=SCHEMA_VERSION,
        expected_artifact_kind=ARTIFACT_KIND,
        expected_schema_sha256=SCHEMA_SHA256,
        expected_schema_id=SCHEMA_ID,
        retained_schema=schemas.preregistration,
    )


def preregistration_fixture_chain_id(
    model: dict[str, Any], *, schemas: ReceiptValidationSchemas
) -> str:
    verify_preregistration_receipt_fixture_model(model, schemas=schemas)
    value = model["phase_chain"]["chain_id_sha256"]
    assert isinstance(value, str)
    return value


def _verify_schema_input(
    schemas: ReceiptValidationSchemas,
) -> RetainedSchemaInput:
    if not isinstance(schemas, ReceiptValidationSchemas):
        raise ImmutableArtifactError("receipt validation schemas are invalid")
    schema = schemas.preregistration
    verify_retained_schema(
        schema.schema_bytes,
        schema.schema_snapshot,
        expected_sha256=SCHEMA_SHA256,
        expected_id=SCHEMA_ID,
    )
    return schema


def _unvalidated_artifact_fixtures_model(
    *,
    target_free_method_artifact: ArtifactIdentity,
    source_inventory_artifact: ArtifactIdentity,
    ancestry_audit_artifact: ArtifactIdentity,
) -> dict[str, Any]:
    identities = (
        (
            target_free_method_artifact,
            METHOD_ARTIFACT_KIND,
            "target-free method",
        ),
        (source_inventory_artifact, SOURCE_ARTIFACT_KIND, "source inventory"),
        (
            ancestry_audit_artifact,
            ANCESTRY_AUDIT_ARTIFACT_KIND,
            "ancestry audit",
        ),
    )
    for identity, expected_kind, label in identities:
        if not isinstance(identity, ArtifactIdentity):
            raise ImmutableArtifactError(f"{label} identity type is invalid")
        require_exact_artifact_kind(identity, expected_kind, label=label)
    return {
        "target_free_method": target_free_method_artifact.to_model(),
        "source_inventory": source_inventory_artifact.to_model(),
        "ancestry_audit": ancestry_audit_artifact.to_model(),
    }


def _parse_unvalidated_artifact_fixtures(value: object) -> dict[str, Any]:
    require_exact_keys(
        value,
        {"target_free_method", "source_inventory", "ancestry_audit"},
        "target-free artifact bindings",
    )
    assert isinstance(value, dict)
    method = ArtifactIdentity.from_model(
        value["target_free_method"], label="target-free method identity"
    )
    source = ArtifactIdentity.from_model(
        value["source_inventory"], label="source inventory identity"
    )
    audit = ArtifactIdentity.from_model(
        value["ancestry_audit"], label="ancestry audit identity"
    )
    return _unvalidated_artifact_fixtures_model(
        target_free_method_artifact=method,
        source_inventory_artifact=source,
        ancestry_audit_artifact=audit,
    )


def _validate_fixture_candidate_inventory(value: object) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "inventory_schema_version",
            "selection_state",
            "candidate_count",
            "candidates",
            "inventory_sha256",
        },
        "fixture candidate inventory",
    )
    assert isinstance(value, dict)
    if type(value["inventory_schema_version"]) is not int or value[
        "inventory_schema_version"
    ] != 1:
        raise ImmutableArtifactError("fixture candidate inventory version is invalid")
    if value["selection_state"] != "FIXTURE_CANDIDATES_COMPLETE_NONAUTHORITATIVE":
        raise ImmutableArtifactError("fixture candidate inventory state is invalid")
    candidates = value["candidates"]
    if type(candidates) is not list or not candidates:
        raise ImmutableArtifactError("fixture candidate inventory is empty or invalid")
    if type(value["candidate_count"]) is not int or value["candidate_count"] != len(
        candidates
    ):
        raise ImmutableArtifactError("fixture candidate inventory count mismatch")
    candidate_ids: list[str] = []
    identity_ids: list[str] = []
    for candidate in candidates:
        require_exact_keys(
            candidate,
            {
                "identity_schema_version",
                "authority_status",
                "candidate_id",
                "canonical_launch_identity_sha256",
                "candidate_record_sha256",
                "candidate_identity_sha256",
            },
            "fixture candidate identity",
        )
        assert isinstance(candidate, dict)
        if (
            type(candidate["identity_schema_version"]) is not int
            or candidate["identity_schema_version"] != 1
        ):
            raise ImmutableArtifactError("fixture candidate identity version is invalid")
        if candidate["authority_status"] != "FIXTURE_ONLY_NONAUTHORITATIVE":
            raise ImmutableArtifactError("fixture candidate authority status is invalid")
        require_ascii_token(candidate["candidate_id"], "fixture candidate_id")
        require_sha256(
            candidate["canonical_launch_identity_sha256"],
            "fixture candidate canonical launch SHA-256",
        )
        require_sha256(
            candidate["candidate_record_sha256"],
            "fixture candidate record SHA-256",
        )
        require_sha256(
            candidate["candidate_identity_sha256"],
            "fixture candidate identity SHA-256",
        )
        identity_core = {
            "identity_schema_version": candidate["identity_schema_version"],
            "authority_status": candidate["authority_status"],
            "candidate_id": candidate["candidate_id"],
            "canonical_launch_identity_sha256": candidate[
                "canonical_launch_identity_sha256"
            ],
            "candidate_record_sha256": candidate["candidate_record_sha256"],
        }
        if candidate["candidate_identity_sha256"] != derive_domain_sha256(
            "FIXTURE_COHORT_CANDIDATE_IDENTITY_V1", identity_core
        ):
            raise ImmutableArtifactError("fixture candidate identity SHA-256 mismatch")
        identity_ids.append(candidate["candidate_identity_sha256"])
        candidate_ids.append(candidate["candidate_id"])
    if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ImmutableArtifactError("fixture candidate inventory is not uniquely ordered")
    if len(identity_ids) != len(set(identity_ids)):
        raise ImmutableArtifactError("fixture candidate inventory duplicates an identity")
    core = {
        "inventory_schema_version": value["inventory_schema_version"],
        "selection_state": value["selection_state"],
        "candidate_count": value["candidate_count"],
        "candidates": candidates,
    }
    expected = derive_domain_sha256("FIXTURE_COHORT_CANDIDATE_INVENTORY_V1", core)
    if value["inventory_sha256"] != expected:
        raise ImmutableArtifactError("fixture candidate inventory SHA-256 mismatch")
    return dict(value)


def _derive_chain_id(
    artifacts: dict[str, Any],
    candidate_inventory: dict[str, Any],
    authority_slots: object,
) -> str:
    return derive_domain_sha256(
        "FIXTURE_EVALUATOR_RECEIPT_CHAIN_V1",
        {
            "unvalidated_artifact_fixtures": artifacts,
            "fixture_candidate_inventory": candidate_inventory,
            "authority_slots": authority_slots,
        },
    )
