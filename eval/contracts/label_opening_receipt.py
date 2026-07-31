"""Closed v1 fixture for a blocked label-opening transition.

Only an explicitly unadmitted opaque fixture digest is bound. Label payloads and
metric values are not accepted by this module, and every production path blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable

from tools.local_labels.immutable_artifact import ImmutableArtifactError

from ._common import (
    AuthorityBlockerCode,
    authority_unavailable,
    evaluation_boundary_model,
    require_exact_evaluation_boundary,
)
from ._receipt_common import (
    ReceiptPhase,
    ReceiptValidationSchemas,
    RetainedSchemaInput,
    derive_domain_sha256,
    phase_chain_model,
    require_ascii_token,
    require_exact_keys,
    require_exact_phase_chain,
    require_sha256,
)
from .model_seal import (
    seal_evaluator_model,
    strict_json_object,
    verify_evaluator_model,
    verify_retained_schema,
)
from .prediction_seal_receipt import (
    prediction_seal_fixture_identity,
    verify_prediction_seal_receipt_fixture_model,
)
from .preregistration_receipt import (
    preregistration_fixture_chain_id,
    verify_preregistration_receipt_fixture_model,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "EVALUATOR_LABEL_OPENING_BLOCKED_FIXTURE"
SCHEMA_ID = (
    "https://gpu-simulator.invalid/eval/schemas/"
    "label_opening_receipt.v1.schema.json"
)
SCHEMA_SHA256 = "5ad671ed6b2c0e3071186cc46a2dc71fd53d7abc9ae3b01acc0fe6b590d92add"
FIXTURE_CONTRACT_MODE = "FIXTURE_ONLY"
LABEL_AUTHORITY_PINS = MappingProxyType(
    {
        "sidecar_schema_version": 5,
        "sidecar_schema_id": "simbench_bw/schemas/local_label_sidecar.schema.json",
        "sidecar_schema_sha256": (
            "15c7111f9ccbff9572c02655e535a9e36305d9681f65a18d072bcdbd4beb3c2e"
        ),
        "admitted_schema_version": 5,
        "admitted_schema_id": "simbench_bw/schemas/local_label_admitted.schema.json",
        "admitted_schema_sha256": (
            "db0bc46f9361344b11f27b6157f3f873bfff7b14c8a049d343b7ed0c0eb13de1"
        ),
        "metric_registry_version": 2,
        "metric_registry_sha256": (
            "95d90f8c4fffd9376b8d785047a9e02ecafc81ed43da6081985958c2b9dec62d"
        ),
    }
)
PRODUCTION_LABEL_ADMISSION_AUTHORITIES = MappingProxyType({})
PRODUCTION_REVIEW_OPERATOR_TRUST_AUTHORITIES = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class FixtureOpaqueUnadmittedLabelBundleIdentity:
    """Opaque fixture binding that explicitly carries no admission claim."""

    bundle_id: str
    bundle_sha256: str
    candidate_inventory_sha256: str
    expected_row_inventory_sha256: str
    fixture_binding_sha256: str
    bundle_schema_version: int = 5
    binding_schema_version: int = 1
    binding_kind: str = "FIXTURE_OPAQUE_UNADMITTED_LABEL_BUNDLE_DIGEST"
    payload_visibility: str = "OPAQUE_FIXTURE_DIGEST_ONLY"
    admission_status: str = "UNAVAILABLE"

    def __post_init__(self) -> None:
        if type(self.binding_schema_version) is not int or self.binding_schema_version != 1:
            raise ImmutableArtifactError("label-bundle binding schema version is unsupported")
        if self.binding_kind != "FIXTURE_OPAQUE_UNADMITTED_LABEL_BUNDLE_DIGEST":
            raise ImmutableArtifactError("fixture label-bundle binding kind is invalid")
        if self.payload_visibility != "OPAQUE_FIXTURE_DIGEST_ONLY":
            raise ImmutableArtifactError("fixture label-bundle payload must remain opaque")
        if self.admission_status != "UNAVAILABLE":
            raise ImmutableArtifactError("fixture label bundle cannot claim admission")
        if type(self.bundle_schema_version) is not int or self.bundle_schema_version != 5:
            raise ImmutableArtifactError("label bundle schema version is unsupported")
        require_ascii_token(self.bundle_id, "bundle_id")
        for value, label in (
            (self.bundle_sha256, "bundle_sha256"),
            (self.candidate_inventory_sha256, "candidate_inventory_sha256"),
            (self.expected_row_inventory_sha256, "expected_row_inventory_sha256"),
            (self.fixture_binding_sha256, "fixture_binding_sha256"),
        ):
            require_sha256(value, label)
        if self.fixture_binding_sha256 != self._derived_sha256():
            raise ImmutableArtifactError("fixture label-bundle binding SHA-256 mismatch")

    @classmethod
    def create(
        cls,
        *,
        bundle_id: str,
        bundle_sha256: str,
        candidate_inventory_sha256: str,
        expected_row_inventory_sha256: str,
    ) -> "FixtureOpaqueUnadmittedLabelBundleIdentity":
        core = {
            "binding_schema_version": 1,
            "binding_kind": "FIXTURE_OPAQUE_UNADMITTED_LABEL_BUNDLE_DIGEST",
            "bundle_schema_version": 5,
            "bundle_id": bundle_id,
            "bundle_sha256": bundle_sha256,
            "candidate_inventory_sha256": candidate_inventory_sha256,
            "expected_row_inventory_sha256": expected_row_inventory_sha256,
            "payload_visibility": "OPAQUE_FIXTURE_DIGEST_ONLY",
            "admission_status": "UNAVAILABLE",
            "label_authority_pins": dict(LABEL_AUTHORITY_PINS),
        }
        return cls(
            bundle_id=bundle_id,
            bundle_sha256=bundle_sha256,
            candidate_inventory_sha256=candidate_inventory_sha256,
            expected_row_inventory_sha256=expected_row_inventory_sha256,
            fixture_binding_sha256=derive_domain_sha256(
                "FIXTURE_UNADMITTED_LABEL_BUNDLE_BINDING_V1", core
            ),
        )

    def _core_model(self) -> dict[str, Any]:
        return {
            "binding_schema_version": self.binding_schema_version,
            "binding_kind": self.binding_kind,
            "bundle_schema_version": self.bundle_schema_version,
            "bundle_id": self.bundle_id,
            "bundle_sha256": self.bundle_sha256,
            "candidate_inventory_sha256": self.candidate_inventory_sha256,
            "expected_row_inventory_sha256": self.expected_row_inventory_sha256,
            "payload_visibility": self.payload_visibility,
            "admission_status": self.admission_status,
            "label_authority_pins": dict(LABEL_AUTHORITY_PINS),
        }

    def _derived_sha256(self) -> str:
        return derive_domain_sha256(
            "FIXTURE_UNADMITTED_LABEL_BUNDLE_BINDING_V1", self._core_model()
        )

    def to_model(self) -> dict[str, Any]:
        return {
            **self._core_model(),
            "fixture_binding_sha256": self.fixture_binding_sha256,
        }

    @classmethod
    def from_model(
        cls, value: object
    ) -> "FixtureOpaqueUnadmittedLabelBundleIdentity":
        require_exact_keys(
            value,
            {
                "binding_schema_version",
                "binding_kind",
                "bundle_schema_version",
                "bundle_id",
                "bundle_sha256",
                "candidate_inventory_sha256",
                "expected_row_inventory_sha256",
                "payload_visibility",
                "admission_status",
                "label_authority_pins",
                "fixture_binding_sha256",
            },
            "fixture opaque unadmitted-label bundle identity",
        )
        assert isinstance(value, dict)
        if value["label_authority_pins"] != LABEL_AUTHORITY_PINS:
            raise ImmutableArtifactError("active v5 label authority pins mismatch")
        return cls(
            binding_schema_version=value["binding_schema_version"],
            binding_kind=value["binding_kind"],
            bundle_schema_version=value["bundle_schema_version"],
            bundle_id=value["bundle_id"],
            bundle_sha256=value["bundle_sha256"],
            candidate_inventory_sha256=value["candidate_inventory_sha256"],
            expected_row_inventory_sha256=value["expected_row_inventory_sha256"],
            payload_visibility=value["payload_visibility"],
            admission_status=value["admission_status"],
            fixture_binding_sha256=value["fixture_binding_sha256"],
        )


def blocked_label_transition_model() -> dict[str, str]:
    return {
        "transition": "BLOCKED_NO_PRODUCTION_LABEL_OPENING",
        "ordering_basis": "FIXTURE_PREDECESSOR_HASH_AND_PHASE_ORDINAL",
        "timestamp_basis": "NOT_USED",
        "label_payload_state": "ABSENT",
        "label_admission": "UNAVAILABLE",
        "scoring": "DISABLED",
    }


def build_label_opening_receipt(**_inputs: Any) -> dict[str, Any]:
    raise authority_unavailable(
        AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
        "no production label admission receipt or trust authority is installed",
    )


def build_label_opening_receipt_fixture(
    *,
    preregistration_receipt: dict[str, Any],
    prediction_seal_receipt: dict[str, Any],
    canonical_crosswalks: Iterable[dict[str, Any]],
    unadmitted_label_bundle_fixture: FixtureOpaqueUnadmittedLabelBundleIdentity,
    schemas: ReceiptValidationSchemas,
) -> dict[str, Any]:
    """Pure fixture builder requiring an already valid exact prediction seal."""

    _verify_schema_input(schemas)
    predecessor = prediction_seal_fixture_identity(
        prediction_seal_receipt,
        preregistration_receipt=preregistration_receipt,
        canonical_crosswalks=canonical_crosswalks,
        schemas=schemas,
    )
    if not isinstance(
        unadmitted_label_bundle_fixture,
        FixtureOpaqueUnadmittedLabelBundleIdentity,
    ):
        raise ImmutableArtifactError("unadmitted fixture label-bundle type is invalid")
    candidate_inventory_sha256 = prediction_seal_receipt[
        "cohort_candidate_inventory"
    ]["inventory_sha256"]
    expected_row_inventory_sha256 = prediction_seal_receipt[
        "expected_row_inventory"
    ]["inventory_sha256"]
    if (
        unadmitted_label_bundle_fixture.candidate_inventory_sha256
        != candidate_inventory_sha256
        or unadmitted_label_bundle_fixture.expected_row_inventory_sha256
        != expected_row_inventory_sha256
    ):
        raise ImmutableArtifactError(
            "fixture label bundle does not bind prediction candidate/row inventories"
        )
    chain_id = preregistration_fixture_chain_id(
        preregistration_receipt, schemas=schemas
    )
    return seal_evaluator_model(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "schema_sha256": SCHEMA_SHA256,
            "contract_mode": FIXTURE_CONTRACT_MODE,
            "production_eligibility": {
                "status": "UNAVAILABLE",
                "reason_codes": [
                    "LABEL_ADMISSION_AUTHORITY_UNAVAILABLE",
                    "REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE",
                ],
            },
            "evaluation_boundary": evaluation_boundary_model(),
            "phase_chain": phase_chain_model(
                phase=ReceiptPhase.LABEL_OPENING,
                chain_id_sha256=chain_id,
                predecessor=predecessor,
            ),
            "unadmitted_label_bundle_fixture": (
                unadmitted_label_bundle_fixture.to_model()
            ),
            "blocked_transition": blocked_label_transition_model(),
        }
    )


def verify_label_opening_receipt_model(
    model: dict[str, Any],
    **_inputs: Any,
) -> None:
    del model
    raise authority_unavailable(
        AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
        "production label opening is blocked by absent admission/trust authorities",
    )


def verify_label_opening_receipt_fixture_model(
    model: dict[str, Any],
    *,
    preregistration_receipt: dict[str, Any],
    prediction_seal_receipt: dict[str, Any],
    canonical_crosswalks: Iterable[dict[str, Any]],
    schemas: ReceiptValidationSchemas,
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
            "unadmitted_label_bundle_fixture",
            "blocked_transition",
            "model_seal_sha256",
        },
        "label-opening receipt",
    )
    if model["contract_mode"] != FIXTURE_CONTRACT_MODE:
        raise ImmutableArtifactError("label-opening blocker is not fixture-only")
    if model["production_eligibility"] != {
        "status": "UNAVAILABLE",
        "reason_codes": [
            "LABEL_ADMISSION_AUTHORITY_UNAVAILABLE",
            "REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE",
        ],
    }:
        raise ImmutableArtifactError("label-opening production blocker mismatch")
    require_exact_evaluation_boundary(model["evaluation_boundary"])
    predecessor = prediction_seal_fixture_identity(
        prediction_seal_receipt,
        preregistration_receipt=preregistration_receipt,
        canonical_crosswalks=canonical_crosswalks,
        schemas=schemas,
    )
    chain_id = preregistration_fixture_chain_id(
        preregistration_receipt, schemas=schemas
    )
    require_exact_phase_chain(
        model["phase_chain"],
        phase=ReceiptPhase.LABEL_OPENING,
        chain_id_sha256=chain_id,
        predecessor=predecessor,
    )
    bundle = FixtureOpaqueUnadmittedLabelBundleIdentity.from_model(
        model["unadmitted_label_bundle_fixture"]
    )
    if (
        bundle.candidate_inventory_sha256
        != prediction_seal_receipt["cohort_candidate_inventory"]["inventory_sha256"]
        or bundle.expected_row_inventory_sha256
        != prediction_seal_receipt["expected_row_inventory"]["inventory_sha256"]
    ):
        raise ImmutableArtifactError(
            "fixture label bundle does not bind prediction candidate/row inventories"
        )
    if model["blocked_transition"] != blocked_label_transition_model():
        raise ImmutableArtifactError("blocked label-opening transition mismatch")


def load_label_opening_receipt(
    payload: bytes,
    **_inputs: Any,
) -> dict[str, Any]:
    del payload
    raise authority_unavailable(
        AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
        "production label-opening loading is blocked",
    )


def load_label_opening_receipt_fixture(
    payload: bytes,
    *,
    preregistration_receipt: dict[str, Any],
    prediction_seal_receipt: dict[str, Any],
    canonical_crosswalks: Iterable[dict[str, Any]],
    schemas: ReceiptValidationSchemas,
) -> dict[str, Any]:
    model = strict_json_object(payload, label="evaluator label-opening blocked fixture")
    verify_label_opening_receipt_fixture_model(
        model,
        preregistration_receipt=preregistration_receipt,
        prediction_seal_receipt=prediction_seal_receipt,
        canonical_crosswalks=canonical_crosswalks,
        schemas=schemas,
    )
    return model


def verify_receipt_chain(**_inputs: Any) -> None:
    raise authority_unavailable(
        AuthorityBlockerCode.REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE,
        "production receipt chain is blocked while review/operator/trust are unresolved",
    )


def verify_receipt_fixture_chain(
    *,
    preregistration_receipt: dict[str, Any],
    prediction_seal_receipt: dict[str, Any],
    label_opening_receipt: dict[str, Any] | None,
    canonical_crosswalks: Iterable[dict[str, Any]],
    schemas: ReceiptValidationSchemas,
) -> None:
    """Verify the exact chain; ``None`` is the only valid pre-opening label state."""

    crosswalks = tuple(canonical_crosswalks)
    verify_preregistration_receipt_fixture_model(
        preregistration_receipt, schemas=schemas
    )
    verify_prediction_seal_receipt_fixture_model(
        prediction_seal_receipt,
        preregistration_receipt=preregistration_receipt,
        canonical_crosswalks=crosswalks,
        schemas=schemas,
    )
    if label_opening_receipt is not None:
        verify_label_opening_receipt_fixture_model(
            label_opening_receipt,
            preregistration_receipt=preregistration_receipt,
            prediction_seal_receipt=prediction_seal_receipt,
            canonical_crosswalks=crosswalks,
            schemas=schemas,
        )


def _verify_schema_input(
    schemas: ReceiptValidationSchemas,
) -> RetainedSchemaInput:
    if not isinstance(schemas, ReceiptValidationSchemas):
        raise ImmutableArtifactError("receipt validation schemas are invalid")
    schema = schemas.label_opening
    verify_retained_schema(
        schema.schema_bytes,
        schema.schema_snapshot,
        expected_sha256=SCHEMA_SHA256,
        expected_id=SCHEMA_ID,
    )
    return schema
