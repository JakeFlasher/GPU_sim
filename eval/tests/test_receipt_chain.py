from __future__ import annotations

import copy
import json
import stat
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
)
from eval.contracts._receipt_common import (
    ArtifactIdentity,
    ReceiptValidationSchemas,
    RetainedSchemaInput,
    derive_domain_sha256,
)
from eval.contracts.label_opening_receipt import (
    LABEL_AUTHORITY_PINS,
    PRODUCTION_LABEL_ADMISSION_AUTHORITIES,
    PRODUCTION_REVIEW_OPERATOR_TRUST_AUTHORITIES,
    FixtureOpaqueUnadmittedLabelBundleIdentity,
    build_label_opening_receipt,
    build_label_opening_receipt_fixture,
    load_label_opening_receipt,
    load_label_opening_receipt_fixture,
    verify_label_opening_receipt_fixture_model,
    verify_label_opening_receipt_model,
    verify_receipt_chain,
    verify_receipt_fixture_chain,
)
from eval.contracts.launch_crosswalk import (
    FixtureLaunchSpec,
    IndependentLaunchBinding,
    LaunchNamespace,
    LaunchRecord,
    LaunchSignature,
    SourceArtifactDescriptor,
    create_launch_inventory_fixture,
    evaluate_launch_crosswalk_fixture,
    fixture_launch_source_payload,
)
from eval.contracts.model_seal import (
    RetainedPayloadSnapshot,
    canonical_model_bytes,
    seal_evaluator_model,
)
from eval.contracts.prediction_seal_receipt import (
    PRODUCTION_COHORT_AUTHORITIES,
    PRODUCTION_SIMULATOR_OUTPUT_AUTHORITIES,
    CohortCandidateIdentity,
    ExpectedPredictionRowIdentity,
    FixtureSimulatorOutputManifest,
    SimulatorArtifactDigests,
    build_prediction_seal_receipt,
    build_prediction_seal_receipt_fixture,
    fixture_candidate_inventory_model,
    load_prediction_seal_receipt,
    load_prediction_seal_receipt_fixture,
    verify_prediction_seal_receipt_fixture_model,
    verify_prediction_seal_receipt_model,
)
from eval.contracts.preregistration_receipt import (
    PRODUCTION_ANCESTRY_REVIEW_AUTHORITIES,
    PRODUCTION_TARGET_FREE_ARTIFACT_AUTHORITIES,
    ANCESTRY_AUDIT_ARTIFACT_KIND,
    METHOD_ARTIFACT_KIND,
    SOURCE_ARTIFACT_KIND,
    build_preregistration_receipt,
    build_preregistration_receipt_fixture,
    load_preregistration_receipt,
    load_preregistration_receipt_fixture,
    verify_preregistration_receipt_fixture_model,
    verify_preregistration_receipt_model,
)
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATHS = {
    "preregistration": ROOT
    / "eval"
    / "schemas"
    / "preregistration_receipt.v1.schema.json",
    "prediction": ROOT
    / "eval"
    / "schemas"
    / "prediction_seal_receipt.v1.schema.json",
    "label_opening": ROOT
    / "eval"
    / "schemas"
    / "label_opening_receipt.v1.schema.json",
    "launch_crosswalk": ROOT
    / "eval"
    / "schemas"
    / "launch_crosswalk.v1.schema.json",
}


def _retained_schema(path: Path) -> RetainedSchemaInput:
    payload, snapshot = RetainedPayloadSnapshot.read_path(path)
    return RetainedSchemaInput(
        schema_bytes=payload,
        schema_snapshot=snapshot,
    )


_CROSSWALK_SCHEMA = _retained_schema(SCHEMA_PATHS["launch_crosswalk"])
SCHEMAS = ReceiptValidationSchemas(
    preregistration=_retained_schema(SCHEMA_PATHS["preregistration"]),
    prediction_seal=_retained_schema(SCHEMA_PATHS["prediction"]),
    label_opening=_retained_schema(SCHEMA_PATHS["label_opening"]),
    launch_crosswalk=_CROSSWALK_SCHEMA,
    launch_inventory=_CROSSWALK_SCHEMA,
)
_SOURCE_KINDS = {
    LaunchNamespace.CANONICAL: "CANONICAL_LAUNCH_SOURCE",
    LaunchNamespace.NCU: "NCU_LAUNCH_SOURCE",
    LaunchNamespace.NVBIT: "NVBIT_LAUNCH_SOURCE",
    LaunchNamespace.SIMULATOR: "SIMULATOR_LAUNCH_SOURCE",
}
SIGNATURE = LaunchSignature(
    kernel_name="kernel_main",
    grid_dim=(8, 2, 1),
    block_dim=(128, 1, 1),
)


def _hash(number: int) -> str:
    return f"{number:064x}"


def _source(
    namespace: LaunchNamespace,
) -> tuple[SourceArtifactDescriptor, bytes]:
    role = "TARGET" if namespace is LaunchNamespace.CANONICAL else "COLD"
    payload = fixture_launch_source_payload(
        namespace,
        (FixtureLaunchSpec(SIGNATURE, 0, role),),
    )
    metadata = (
        1,
        100 + tuple(LaunchNamespace).index(namespace),
        stat.S_IFREG | 0o600,
        1,
        len(payload),
        1,
        1,
    )
    snapshot = RetainedPayloadSnapshot.from_fixture_payload(payload, metadata)
    return (
        SourceArtifactDescriptor.from_fixture_payload(
            namespace=namespace,
            artifact_kind=_SOURCE_KINDS[namespace],
            payload=payload,
            snapshot=snapshot,
        ),
        payload,
    )


def _record(source: SourceArtifactDescriptor) -> LaunchRecord:
    return LaunchRecord.create(
        source=source,
        signature=SIGNATURE,
        launch_ordinal=0,
        launch_role=(
            "TARGET"
            if source.namespace is LaunchNamespace.CANONICAL
            else "COLD"
        ),
    )


def _crosswalk() -> dict[str, Any]:
    sources: dict[LaunchNamespace, SourceArtifactDescriptor] = {}
    payloads: dict[LaunchNamespace, bytes] = {}
    records: dict[LaunchNamespace, tuple[LaunchRecord, ...]] = {}
    inventories: dict[LaunchNamespace, dict[str, Any]] = {}
    for namespace in LaunchNamespace:
        sources[namespace], payloads[namespace] = _source(namespace)
        records[namespace] = (_record(sources[namespace]),)
        inventories[namespace] = create_launch_inventory_fixture(
            source=sources[namespace],
            source_payload=payloads[namespace],
            candidates=records[namespace],
            schema_bytes=SCHEMAS.launch_inventory.schema_bytes,
            schema_snapshot=SCHEMAS.launch_inventory.schema_snapshot,
        )
    bindings = {
        namespace: IndependentLaunchBinding.create_fixture(
            inventory=inventories[namespace],
            canonical_launch=records[LaunchNamespace.CANONICAL][0],
            selected_launch=records[namespace][0],
            inventory_schema_bytes=SCHEMAS.launch_inventory.schema_bytes,
            inventory_schema_snapshot=SCHEMAS.launch_inventory.schema_snapshot,
        )
        for namespace in (
            LaunchNamespace.NCU,
            LaunchNamespace.NVBIT,
            LaunchNamespace.SIMULATOR,
        )
    }
    return evaluate_launch_crosswalk_fixture(
        canonical_inventory=inventories[LaunchNamespace.CANONICAL],
        ncu_inventory=inventories[LaunchNamespace.NCU],
        nvbit_inventory=inventories[LaunchNamespace.NVBIT],
        simulator_inventory=inventories[LaunchNamespace.SIMULATOR],
        canonical_candidates=records[LaunchNamespace.CANONICAL],
        ncu_candidates=records[LaunchNamespace.NCU],
        nvbit_candidates=records[LaunchNamespace.NVBIT],
        simulator_candidates=records[LaunchNamespace.SIMULATOR],
        ncu_binding=bindings[LaunchNamespace.NCU],
        nvbit_binding=bindings[LaunchNamespace.NVBIT],
        simulator_binding=bindings[LaunchNamespace.SIMULATOR],
        schema_bytes=SCHEMAS.launch_crosswalk.schema_bytes,
        schema_snapshot=SCHEMAS.launch_crosswalk.schema_snapshot,
        inventory_schema_bytes=SCHEMAS.launch_inventory.schema_bytes,
        inventory_schema_snapshot=SCHEMAS.launch_inventory.schema_snapshot,
    )


def _artifact(kind: str, artifact_id: str, number: int) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_kind=kind,
        artifact_schema_version=1,
        artifact_id=artifact_id,
        artifact_sha256=_hash(number),
    )


def _preregistration(
    crosswalk: dict[str, Any],
    *,
    method_hash_number: int = 20,
    candidate_record_number: int = 30,
) -> dict[str, Any]:
    candidates, _ = _candidate_rows(
        crosswalk, candidate_record_number=candidate_record_number
    )
    candidate_inventory = fixture_candidate_inventory_model(
        candidates,
        crosswalk_inventory={
            "entries": [
                {
                    "canonical_launch_identity_sha256": candidates[
                        0
                    ].canonical_launch_identity_sha256
                }
            ]
        },
    )
    return build_preregistration_receipt_fixture(
        target_free_method_artifact=_artifact(
            METHOD_ARTIFACT_KIND, "method.v1", method_hash_number
        ),
        source_inventory_artifact=ArtifactIdentity(
            artifact_kind=SOURCE_ARTIFACT_KIND,
            artifact_schema_version=1,
            artifact_id="sources.v1",
            artifact_sha256=sha256_bytes(canonical_json_bytes(candidate_inventory)),
        ),
        ancestry_audit_artifact=_artifact(
            ANCESTRY_AUDIT_ARTIFACT_KIND, "ancestry.audit.v1", 22
        ),
        fixture_candidate_inventory=candidate_inventory,
        schemas=SCHEMAS,
    )


def _candidate_rows(
    crosswalk: dict[str, Any],
    *,
    row_hash_number: int = 40,
    candidate_record_number: int = 30,
) -> tuple[tuple[CohortCandidateIdentity, ...], tuple[ExpectedPredictionRowIdentity, ...]]:
    canonical_sha = crosswalk["namespaces"]["canonical"]["selected_launch"][
        "identity"
    ]["qualified_launch_sha256"]
    candidates = (
        CohortCandidateIdentity.create(
            candidate_id="candidate.a",
            canonical_launch_identity_sha256=canonical_sha,
            candidate_record_sha256=_hash(candidate_record_number),
        ),
        CohortCandidateIdentity.create(
            candidate_id="candidate.b",
            canonical_launch_identity_sha256=canonical_sha,
            candidate_record_sha256=_hash(candidate_record_number + 1),
        ),
    )
    rows = tuple(
        ExpectedPredictionRowIdentity.create(
            row_id=candidate.candidate_id,
            cohort_candidate_identity_sha256=candidate.candidate_identity_sha256,
            canonical_launch_identity_sha256=candidate.canonical_launch_identity_sha256,
            simulator_output_row_sha256=_hash(row_hash_number + index),
        )
        for index, candidate in enumerate(candidates)
    )
    return candidates, rows


def _prediction(
    preregistration: dict[str, Any],
    crosswalk: dict[str, Any],
    *,
    output_hash_number: int = 40,
    candidate_record_number: int = 30,
    reverse_inputs: bool = False,
) -> dict[str, Any]:
    candidates, rows = _candidate_rows(
        crosswalk,
        row_hash_number=output_hash_number,
        candidate_record_number=candidate_record_number,
    )
    if reverse_inputs:
        candidates = tuple(reversed(candidates))
        rows = tuple(reversed(rows))
    output_manifest = FixtureSimulatorOutputManifest.create(rows)
    return build_prediction_seal_receipt_fixture(
        preregistration_receipt=preregistration,
        canonical_crosswalks=(crosswalk,),
        cohort_candidates=candidates,
        simulator_artifacts=SimulatorArtifactDigests(
            build_sha256=_hash(50),
            config_sha256=_hash(51),
            input_sha256=_hash(52),
            output_sha256=output_manifest.manifest_sha256,
        ),
        simulator_output_manifest=output_manifest,
        expected_rows=rows,
        schemas=SCHEMAS,
    )


def _chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    crosswalk = _crosswalk()
    preregistration = _preregistration(crosswalk)
    prediction = _prediction(preregistration, crosswalk)
    label_fixture = FixtureOpaqueUnadmittedLabelBundleIdentity.create(
        bundle_id="unadmitted.labels.v5",
        bundle_sha256=_hash(60),
        candidate_inventory_sha256=prediction["cohort_candidate_inventory"][
            "inventory_sha256"
        ],
        expected_row_inventory_sha256=prediction["expected_row_inventory"][
            "inventory_sha256"
        ],
    )
    label_opening = build_label_opening_receipt_fixture(
        preregistration_receipt=preregistration,
        prediction_seal_receipt=prediction,
        canonical_crosswalks=(crosswalk,),
        unadmitted_label_bundle_fixture=label_fixture,
        schemas=SCHEMAS,
    )
    return preregistration, prediction, label_opening, crosswalk


def _validator(name: str) -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATHS[name].read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _reseal(model: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(model)
    unsigned.pop("model_seal_sha256", None)
    return seal_evaluator_model(unsigned)


def _recompute_row_inventory_sha(model: dict[str, Any]) -> None:
    inventory = model["expected_row_inventory"]
    core = {
        "inventory_schema_version": inventory["inventory_schema_version"],
        "row_count": inventory["row_count"],
        "rows": inventory["rows"],
    }
    inventory["inventory_sha256"] = derive_domain_sha256(
        "FIXTURE_EXPECTED_PREDICTION_ROW_INVENTORY_V1", core
    )


def _verify_prediction(
    model: dict[str, Any], preregistration: dict[str, Any], crosswalk: dict[str, Any]
) -> None:
    verify_prediction_seal_receipt_fixture_model(
        model,
        preregistration_receipt=preregistration,
        canonical_crosswalks=(crosswalk,),
        schemas=SCHEMAS,
    )


def _verify_label(
    model: dict[str, Any],
    preregistration: dict[str, Any],
    prediction: dict[str, Any],
    crosswalk: dict[str, Any],
) -> None:
    verify_label_opening_receipt_fixture_model(
        model,
        preregistration_receipt=preregistration,
        prediction_seal_receipt=prediction,
        canonical_crosswalks=(crosswalk,),
        schemas=SCHEMAS,
    )


def test_production_authority_registries_are_empty_and_paths_block() -> None:
    assert dict(PRODUCTION_TARGET_FREE_ARTIFACT_AUTHORITIES) == {}
    assert dict(PRODUCTION_ANCESTRY_REVIEW_AUTHORITIES) == {}
    assert dict(PRODUCTION_COHORT_AUTHORITIES) == {}
    assert dict(PRODUCTION_SIMULATOR_OUTPUT_AUTHORITIES) == {}
    assert dict(PRODUCTION_LABEL_ADMISSION_AUTHORITIES) == {}
    assert dict(PRODUCTION_REVIEW_OPERATOR_TRUST_AUTHORITIES) == {}

    calls = (
        (
            lambda: build_preregistration_receipt(),
            AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: verify_preregistration_receipt_model({}, schemas=SCHEMAS),
            AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: load_preregistration_receipt(b"{}", schemas=SCHEMAS),
            AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: build_prediction_seal_receipt(),
            AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: verify_prediction_seal_receipt_model({}),
            AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: load_prediction_seal_receipt(b"{}"),
            AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: build_label_opening_receipt(),
            AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: verify_label_opening_receipt_model({}),
            AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: load_label_opening_receipt(b"{}"),
            AuthorityBlockerCode.LABEL_ADMISSION_AUTHORITY_UNAVAILABLE,
        ),
        (
            lambda: verify_receipt_chain(),
            AuthorityBlockerCode.REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE,
        ),
    )
    for call, expected_code in calls:
        with pytest.raises(EvaluatorAuthorityUnavailable) as captured:
            call()
        assert captured.value.code is expected_code


def test_current_v5_label_authority_file_pins_are_exact() -> None:
    sidecar_path = ROOT / "workloads/simbench_bw/schemas/local_label_sidecar.schema.json"
    admitted_path = ROOT / "workloads/simbench_bw/schemas/local_label_admitted.schema.json"
    registry_path = ROOT / "tools/local_labels/metric_registry.json"
    assert sha256_bytes(sidecar_path.read_bytes()) == LABEL_AUTHORITY_PINS[
        "sidecar_schema_sha256"
    ]
    assert sha256_bytes(admitted_path.read_bytes()) == LABEL_AUTHORITY_PINS[
        "admitted_schema_sha256"
    ]
    assert sha256_bytes(registry_path.read_bytes()) == LABEL_AUTHORITY_PINS[
        "metric_registry_sha256"
    ]
    assert json.loads(sidecar_path.read_bytes())["$id"] == LABEL_AUTHORITY_PINS[
        "sidecar_schema_id"
    ]
    assert json.loads(admitted_path.read_bytes())["$id"] == LABEL_AUTHORITY_PINS[
        "admitted_schema_id"
    ]
    assert json.loads(registry_path.read_bytes())["registry_version"] == (
        LABEL_AUTHORITY_PINS["metric_registry_version"]
    )


def test_valid_chain_is_canonical_deterministic_and_schema_valid() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()
    reordered_prediction = _prediction(
        preregistration, crosswalk, reverse_inputs=True
    )

    assert prediction == reordered_prediction
    assert prediction["cycles_prediction"] == {
        "descriptor_id": "exec_cycles_gpc_max",
        "metric_id": "gpc__cycles_elapsed.max",
        "implementation_status": "required_unimplemented",
        "evaluation_state": "pending",
        "outcome": "unavailable",
        "score_status": "unscored",
    }
    assert [
        preregistration["phase_chain"]["phase"],
        prediction["phase_chain"]["phase"],
        label_opening["phase_chain"]["phase"],
    ] == [
        "PREREGISTRATION_FIXTURE",
        "PREDICTION_SEAL_FIXTURE",
        "LABEL_OPENING_BLOCKED_FIXTURE",
    ]
    assert [
        preregistration["phase_chain"]["phase_ordinal"],
        prediction["phase_chain"]["phase_ordinal"],
        label_opening["phase_chain"]["phase_ordinal"],
    ] == [0, 1, 2]
    assert all(
        receipt["contract_mode"] == "FIXTURE_ONLY"
        for receipt in (preregistration, prediction, label_opening)
    )
    assert preregistration["production_eligibility"] == {
        "status": "UNAVAILABLE",
        "reason_codes": [
            "TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE",
            "ANCESTRY_REVIEW_AUTHORITY_UNAVAILABLE",
        ],
    }
    assert prediction["production_eligibility"] == {
        "status": "UNAVAILABLE",
        "reason_codes": [
            "COHORT_AUTHORITY_UNAVAILABLE",
            "SIMULATOR_OUTPUT_AUTHORITY_UNAVAILABLE",
        ],
    }
    assert label_opening["production_eligibility"] == {
        "status": "UNAVAILABLE",
        "reason_codes": [
            "LABEL_ADMISSION_AUTHORITY_UNAVAILABLE",
            "REVIEW_OPERATOR_TRUST_AUTHORITY_UNAVAILABLE",
        ],
    }
    bundle = label_opening["unadmitted_label_bundle_fixture"]
    assert bundle["binding_kind"] == "FIXTURE_OPAQUE_UNADMITTED_LABEL_BUNDLE_DIGEST"
    assert bundle["bundle_schema_version"] == 5
    assert bundle["bundle_id"] == "unadmitted.labels.v5"
    assert bundle["bundle_sha256"] == _hash(60)
    assert bundle["payload_visibility"] == "OPAQUE_FIXTURE_DIGEST_ONLY"
    assert bundle["admission_status"] == "UNAVAILABLE"
    assert bundle["label_authority_pins"] == LABEL_AUTHORITY_PINS
    assert label_opening["blocked_transition"] == {
        "transition": "BLOCKED_NO_PRODUCTION_LABEL_OPENING",
        "ordering_basis": "FIXTURE_PREDECESSOR_HASH_AND_PHASE_ORDINAL",
        "timestamp_basis": "NOT_USED",
        "label_payload_state": "ABSENT",
        "label_admission": "UNAVAILABLE",
        "scoring": "DISABLED",
    }

    verify_receipt_fixture_chain(
        preregistration_receipt=preregistration,
        prediction_seal_receipt=prediction,
        label_opening_receipt=label_opening,
        canonical_crosswalks=(crosswalk,),
        schemas=SCHEMAS,
    )
    _validator("preregistration").validate(preregistration)
    _validator("prediction").validate(prediction)
    _validator("label_opening").validate(label_opening)
    assert (
        load_preregistration_receipt_fixture(
            canonical_model_bytes(preregistration), schemas=SCHEMAS
        )
        == preregistration
    )
    assert (
        load_prediction_seal_receipt_fixture(
            canonical_model_bytes(prediction),
            preregistration_receipt=preregistration,
            canonical_crosswalks=(crosswalk,),
            schemas=SCHEMAS,
        )
        == prediction
    )
    assert (
        load_label_opening_receipt_fixture(
            canonical_model_bytes(label_opening),
            preregistration_receipt=preregistration,
            prediction_seal_receipt=prediction,
            canonical_crosswalks=(crosswalk,),
            schemas=SCHEMAS,
        )
        == label_opening
    )


def test_receipts_have_no_label_values_target_values_or_scoring_surface() -> None:
    receipts = _chain()[:3]
    forbidden_keys = {
        "metric_value",
        "metric_values",
        "observed_value",
        "predicted_value",
        "target_value",
        "target_values",
        "label_value",
        "label_values",
        "score",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for receipt in receipts:
        visit(receipt)
        assert receipt["evaluation_boundary"]["cycles"]["outcome"] == "unavailable"


def test_duplicate_json_keys_are_rejected_at_root_and_nested_depth() -> None:
    preregistration, prediction, _, crosswalk = _chain()
    root_duplicate = (
        canonical_model_bytes(preregistration)[:-1]
        + b',"schema_version":1}'
    )
    with pytest.raises(ImmutableArtifactError, match="duplicate JSON key"):
        load_preregistration_receipt_fixture(root_duplicate, schemas=SCHEMAS)

    payload = canonical_model_bytes(prediction)
    nested_duplicate = payload.replace(
        b'"row_id":"candidate.a"',
        b'"row_id":"candidate.a","row_id":"candidate.a"',
        1,
    )
    with pytest.raises(ImmutableArtifactError, match="duplicate JSON key"):
        load_prediction_seal_receipt_fixture(
            nested_duplicate,
            preregistration_receipt=preregistration,
            canonical_crosswalks=(crosswalk,),
            schemas=SCHEMAS,
        )


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity"])
def test_nonfinite_json_is_rejected_before_contract_validation(literal: bytes) -> None:
    payload = b'{"schema_version":1,"probe":' + literal + b"}"
    with pytest.raises(ImmutableArtifactError, match="non-finite"):
        load_preregistration_receipt_fixture(payload, schemas=SCHEMAS)


def test_preregistration_rejects_target_derived_fields_and_authority_population() -> None:
    crosswalk = _crosswalk()
    preregistration = _preregistration(crosswalk)
    validator = _validator("preregistration")

    target_derived = copy.deepcopy(preregistration)
    target_derived["unvalidated_artifact_fixtures"]["target_free_method"][
        "target_metric_id"
    ] = "gpc__cycles_elapsed.max"
    target_derived = _reseal(target_derived)
    with pytest.raises(ImmutableArtifactError, match="fields are not exact"):
        verify_preregistration_receipt_fixture_model(target_derived, schemas=SCHEMAS)
    with pytest.raises(ValidationError):
        validator.validate(target_derived)

    accepted = copy.deepcopy(preregistration)
    accepted["authority_slots"][0]["status"] = "ACCEPTED"
    accepted = _reseal(accepted)
    with pytest.raises(ImmutableArtifactError, match="not unresolved"):
        verify_preregistration_receipt_fixture_model(accepted, schemas=SCHEMAS)
    with pytest.raises(ValidationError):
        validator.validate(accepted)

    unknown = copy.deepcopy(preregistration)
    unknown["authority_slots"].append(
        {"authority": "UNKNOWN_AUTHORITY", "status": "UNRESOLVED"}
    )
    unknown = _reseal(unknown)
    with pytest.raises(ImmutableArtifactError, match="not unresolved"):
        verify_preregistration_receipt_fixture_model(unknown, schemas=SCHEMAS)
    with pytest.raises(ValidationError):
        validator.validate(unknown)


def test_target_bearing_opaque_digest_is_only_an_unvalidated_fixture() -> None:
    crosswalk = _crosswalk()
    base = _preregistration(crosswalk)
    target_bearing_payload = canonical_json_bytes(
        {"target_value": 123, "label_values": [456], "score": 0.01}
    )
    method_fixture = ArtifactIdentity(
        artifact_kind=METHOD_ARTIFACT_KIND,
        artifact_schema_version=1,
        artifact_id="method.target.bearing.fixture",
        artifact_sha256=sha256_bytes(target_bearing_payload),
    )
    source_fixture = ArtifactIdentity.from_model(
        base["unvalidated_artifact_fixtures"]["source_inventory"]
    )
    ancestry_fixture = ArtifactIdentity.from_model(
        base["unvalidated_artifact_fixtures"]["ancestry_audit"]
    )
    receipt = build_preregistration_receipt_fixture(
        target_free_method_artifact=method_fixture,
        source_inventory_artifact=source_fixture,
        ancestry_audit_artifact=ancestry_fixture,
        fixture_candidate_inventory=base["fixture_candidate_inventory"],
        schemas=SCHEMAS,
    )
    verify_preregistration_receipt_fixture_model(receipt, schemas=SCHEMAS)
    assert receipt["production_eligibility"]["status"] == "UNAVAILABLE"
    assert receipt["unvalidated_artifact_fixtures"]["target_free_method"][
        "authority_status"
    ] == "FIXTURE_ONLY_OPAQUE_UNVALIDATED"
    assert receipt["unvalidated_artifact_fixtures"]["target_free_method"][
        "artifact_kind"
    ] == "FIXTURE_UNVALIDATED_EVALUATION_METHOD"
    with pytest.raises(EvaluatorAuthorityUnavailable) as production_blocker:
        verify_preregistration_receipt_model(receipt, schemas=SCHEMAS)
    assert (
        production_blocker.value.code
        is AuthorityBlockerCode.TARGET_FREE_ARTIFACT_AUTHORITY_UNAVAILABLE
    )

    with pytest.raises(ImmutableArtifactError, match="must be exactly 1"):
        ArtifactIdentity(
            artifact_kind=METHOD_ARTIFACT_KIND,
            artifact_schema_version=999,
            artifact_id="method.v999",
            artifact_sha256=sha256_bytes(target_bearing_payload),
        )


def test_preregistered_fixture_candidate_identity_is_recomputed() -> None:
    crosswalk = _crosswalk()
    base = _preregistration(crosswalk)
    hostile_inventory = copy.deepcopy(base["fixture_candidate_inventory"])
    hostile_inventory["candidates"][0]["candidate_record_sha256"] = _hash(888)
    with pytest.raises(
        ImmutableArtifactError,
        match="fixture candidate identity SHA-256 mismatch",
    ):
        build_preregistration_receipt_fixture(
            target_free_method_artifact=ArtifactIdentity.from_model(
                base["unvalidated_artifact_fixtures"]["target_free_method"]
            ),
            source_inventory_artifact=ArtifactIdentity.from_model(
                base["unvalidated_artifact_fixtures"]["source_inventory"]
            ),
            ancestry_audit_artifact=ArtifactIdentity.from_model(
                base["unvalidated_artifact_fixtures"]["ancestry_audit"]
            ),
            fixture_candidate_inventory=hostile_inventory,
            schemas=SCHEMAS,
        )


def test_strict_versions_hashes_unknown_fields_and_bool_as_int() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()

    wrong_version = copy.deepcopy(preregistration)
    wrong_version["schema_version"] = 2
    wrong_version = _reseal(wrong_version)
    with pytest.raises(ImmutableArtifactError, match="schema_version mismatch"):
        verify_preregistration_receipt_fixture_model(wrong_version, schemas=SCHEMAS)
    with pytest.raises(ValidationError):
        _validator("preregistration").validate(wrong_version)

    boolean_count = copy.deepcopy(prediction)
    boolean_count["cohort_candidate_inventory"]["candidate_count"] = True
    boolean_count = _reseal(boolean_count)
    with pytest.raises(ImmutableArtifactError, match="positive|nonnegative|candidate_count"):
        _verify_prediction(boolean_count, preregistration, crosswalk)
    with pytest.raises(ValidationError):
        _validator("prediction").validate(boolean_count)

    boolean_phase_version = copy.deepcopy(prediction)
    boolean_phase_version["phase_chain"]["chain_schema_version"] = True
    boolean_phase_version = _reseal(boolean_phase_version)
    with pytest.raises(ImmutableArtifactError, match="phase/state-lifetime"):
        _verify_prediction(boolean_phase_version, preregistration, crosswalk)
    with pytest.raises(ValidationError):
        _validator("prediction").validate(boolean_phase_version)

    boolean_manifest_version = copy.deepcopy(prediction)
    boolean_manifest_version["fixture_simulator_output_manifest"][
        "manifest_schema_version"
    ] = True
    boolean_manifest_version = _reseal(boolean_manifest_version)
    with pytest.raises(ImmutableArtifactError, match="manifest version"):
        _verify_prediction(boolean_manifest_version, preregistration, crosswalk)
    with pytest.raises(ValidationError):
        _validator("prediction").validate(boolean_manifest_version)

    malformed_hash = copy.deepcopy(label_opening)
    malformed_hash["unadmitted_label_bundle_fixture"]["bundle_sha256"] = "A" * 64
    malformed_hash = _reseal(malformed_hash)
    with pytest.raises(ImmutableArtifactError, match="lowercase 64-hex"):
        _verify_label(malformed_hash, preregistration, prediction, crosswalk)
    with pytest.raises(ValidationError):
        _validator("label_opening").validate(malformed_hash)

    unknown_field = copy.deepcopy(label_opening)
    unknown_field["reviewer_approved"] = False
    unknown_field = _reseal(unknown_field)
    with pytest.raises(ImmutableArtifactError, match="fields are not exact"):
        _verify_label(unknown_field, preregistration, prediction, crosswalk)
    with pytest.raises(ValidationError):
        _validator("label_opening").validate(unknown_field)


def test_phase_hash_and_predecessor_mismatches_fail_closed() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()

    wrong_phase = copy.deepcopy(prediction)
    wrong_phase["phase_chain"]["phase_ordinal"] = 2
    wrong_phase = _reseal(wrong_phase)
    with pytest.raises(ImmutableArtifactError, match="phase/state-lifetime"):
        _verify_prediction(wrong_phase, preregistration, crosswalk)
    with pytest.raises(ValidationError):
        _validator("prediction").validate(wrong_phase)

    wrong_predecessor = copy.deepcopy(label_opening)
    wrong_predecessor["phase_chain"]["predecessor_receipt"][
        "canonical_model_sha256"
    ] = _hash(999)
    wrong_predecessor = _reseal(wrong_predecessor)
    _validator("label_opening").validate(wrong_predecessor)
    with pytest.raises(ImmutableArtifactError, match="phase/state-lifetime"):
        _verify_label(wrong_predecessor, preregistration, prediction, crosswalk)

    wrong_inner_hash = copy.deepcopy(prediction)
    wrong_inner_hash["expected_row_inventory"]["rows"][0][
        "expected_row_identity_sha256"
    ] = _hash(998)
    wrong_inner_hash = _reseal(wrong_inner_hash)
    _validator("prediction").validate(wrong_inner_hash)
    with pytest.raises(ImmutableArtifactError, match="row identity SHA-256 mismatch"):
        _verify_prediction(wrong_inner_hash, preregistration, crosswalk)


def test_timestamp_fields_and_noncanonical_array_order_are_rejected() -> None:
    preregistration, prediction, _, crosswalk = _chain()

    timestamped = copy.deepcopy(preregistration)
    timestamped["phase_chain"]["created_at"] = "2026-07-30T12:00:00Z"
    timestamped = _reseal(timestamped)
    with pytest.raises(ImmutableArtifactError, match="phase/state-lifetime"):
        verify_preregistration_receipt_fixture_model(timestamped, schemas=SCHEMAS)
    with pytest.raises(ValidationError):
        _validator("preregistration").validate(timestamped)

    reordered = copy.deepcopy(prediction)
    reordered["expected_row_inventory"]["rows"].reverse()
    _recompute_row_inventory_sha(reordered)
    reordered = _reseal(reordered)
    _validator("prediction").validate(reordered)
    with pytest.raises(ImmutableArtifactError, match="not canonically ordered"):
        _verify_prediction(reordered, preregistration, crosswalk)


def test_prediction_subset_cannot_redefine_preregistered_completeness() -> None:
    preregistration, _, _, crosswalk = _chain()
    candidates, rows = _candidate_rows(crosswalk)
    subset_rows = (rows[0],)
    subset_manifest = FixtureSimulatorOutputManifest.create(subset_rows)
    with pytest.raises(
        ImmutableArtifactError,
        match="candidate inventory differs from preregistration",
    ):
        build_prediction_seal_receipt_fixture(
            preregistration_receipt=preregistration,
            canonical_crosswalks=(crosswalk,),
            cohort_candidates=(candidates[0],),
            simulator_artifacts=SimulatorArtifactDigests(
                build_sha256=_hash(50),
                config_sha256=_hash(51),
                input_sha256=_hash(52),
                output_sha256=subset_manifest.manifest_sha256,
            ),
            simulator_output_manifest=subset_manifest,
            expected_rows=subset_rows,
            schemas=SCHEMAS,
        )


def test_simulator_output_digest_and_manifest_must_exactly_cover_rows() -> None:
    preregistration, _, _, crosswalk = _chain()
    candidates, rows = _candidate_rows(crosswalk)
    full_manifest = FixtureSimulatorOutputManifest.create(rows)
    with pytest.raises(
        ImmutableArtifactError,
        match="output digest does not equal canonical manifest digest",
    ):
        build_prediction_seal_receipt_fixture(
            preregistration_receipt=preregistration,
            canonical_crosswalks=(crosswalk,),
            cohort_candidates=candidates,
            simulator_artifacts=SimulatorArtifactDigests(
                build_sha256=_hash(50),
                config_sha256=_hash(51),
                input_sha256=_hash(52),
                output_sha256=_hash(999),
            ),
            simulator_output_manifest=full_manifest,
            expected_rows=rows,
            schemas=SCHEMAS,
        )

    subset_manifest = FixtureSimulatorOutputManifest.create((rows[0],))
    with pytest.raises(
        ImmutableArtifactError,
        match="manifest does not exactly cover expected rows",
    ):
        build_prediction_seal_receipt_fixture(
            preregistration_receipt=preregistration,
            canonical_crosswalks=(crosswalk,),
            cohort_candidates=candidates,
            simulator_artifacts=SimulatorArtifactDigests(
                build_sha256=_hash(50),
                config_sha256=_hash(51),
                input_sha256=_hash(52),
                output_sha256=subset_manifest.manifest_sha256,
            ),
            simulator_output_manifest=subset_manifest,
            expected_rows=rows,
            schemas=SCHEMAS,
        )


def test_missing_expected_rows_and_semantic_duplicates_are_rejected() -> None:
    preregistration, prediction, _, crosswalk = _chain()

    missing = copy.deepcopy(prediction)
    missing["expected_row_inventory"]["rows"].pop()
    missing["expected_row_inventory"]["row_count"] = 1
    _recompute_row_inventory_sha(missing)
    missing = _reseal(missing)
    _validator("prediction").validate(missing)
    with pytest.raises(ImmutableArtifactError, match="cover every cohort candidate"):
        _verify_prediction(missing, preregistration, crosswalk)

    candidates, rows = _candidate_rows(crosswalk)
    duplicate_id = CohortCandidateIdentity.create(
        candidate_id=candidates[0].candidate_id,
        canonical_launch_identity_sha256=candidates[0].canonical_launch_identity_sha256,
        candidate_record_sha256=_hash(777),
    )
    output_manifest = FixtureSimulatorOutputManifest.create(rows)
    with pytest.raises(ImmutableArtifactError, match="duplicate cohort-candidate ID"):
        build_prediction_seal_receipt_fixture(
            preregistration_receipt=preregistration,
            canonical_crosswalks=(crosswalk,),
            cohort_candidates=(candidates[0], duplicate_id),
            simulator_artifacts=SimulatorArtifactDigests(
                build_sha256=_hash(50),
                config_sha256=_hash(51),
                input_sha256=_hash(52),
                output_sha256=output_manifest.manifest_sha256,
            ),
            simulator_output_manifest=output_manifest,
            expected_rows=rows,
            schemas=SCHEMAS,
        )


def test_unknown_label_bundle_version_and_cross_candidate_replay_are_rejected() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()
    bundle_model = copy.deepcopy(
        label_opening["unadmitted_label_bundle_fixture"]
    )
    bundle_model["bundle_schema_version"] = 999
    with pytest.raises(ImmutableArtifactError, match="version is unsupported"):
        FixtureOpaqueUnadmittedLabelBundleIdentity.from_model(bundle_model)

    original_bundle = FixtureOpaqueUnadmittedLabelBundleIdentity.from_model(
        label_opening["unadmitted_label_bundle_fixture"]
    )
    alternate_preregistration = _preregistration(
        crosswalk, candidate_record_number=130
    )
    alternate_prediction = _prediction(
        alternate_preregistration,
        crosswalk,
        candidate_record_number=130,
    )
    assert original_bundle.bundle_sha256 == _hash(60)
    assert (
        alternate_prediction["cohort_candidate_inventory"]["inventory_sha256"]
        != prediction["cohort_candidate_inventory"]["inventory_sha256"]
    )
    with pytest.raises(
        ImmutableArtifactError,
        match="does not bind prediction candidate/row inventories",
    ):
        build_label_opening_receipt_fixture(
            preregistration_receipt=alternate_preregistration,
            prediction_seal_receipt=alternate_prediction,
            canonical_crosswalks=(crosswalk,),
            unadmitted_label_bundle_fixture=original_bundle,
            schemas=SCHEMAS,
        )


def test_label_value_smuggling_is_rejected_at_every_closed_surface() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()
    validator = _validator("label_opening")

    for location in ("bundle", "root"):
        smuggled = copy.deepcopy(label_opening)
        if location == "bundle":
            smuggled["unadmitted_label_bundle_fixture"]["metric_values"] = {
                "gpc__cycles_elapsed.max": 123.0
            }
        else:
            smuggled["label_values"] = [123.0]
        smuggled = _reseal(smuggled)
        with pytest.raises(ImmutableArtifactError, match="fields are not exact"):
            _verify_label(smuggled, preregistration, prediction, crosswalk)
        with pytest.raises(ValidationError):
            validator.validate(smuggled)


def test_unicode_normalization_and_path_aliases_are_not_identifiers() -> None:
    invalid_ids = (
        "candidate/../other",
        "candidate\\other",
        "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
        unicodedata.normalize("NFD", "caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
        "candidate..other",
        "C:relative",
        "candidate:other",
    )
    for invalid_id in invalid_ids:
        with pytest.raises(ImmutableArtifactError, match="ASCII token"):
            CohortCandidateIdentity.create(
                candidate_id=invalid_id,
                canonical_launch_identity_sha256=_hash(1),
                candidate_record_sha256=_hash(2),
            )

    with pytest.raises(ImmutableArtifactError, match="ASCII token"):
        _artifact(METHOD_ARTIFACT_KIND, "method/alias", 1)


def test_label_opening_requires_exact_seal_and_rejects_stale_replay() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()
    stale_prediction = _prediction(
        preregistration, crosswalk, output_hash_number=54
    )
    verify_prediction_seal_receipt_fixture_model(
        stale_prediction,
        preregistration_receipt=preregistration,
        canonical_crosswalks=(crosswalk,),
        schemas=SCHEMAS,
    )

    with pytest.raises(ImmutableArtifactError, match="phase/state-lifetime"):
        _verify_label(
            label_opening,
            preregistration,
            stale_prediction,
            crosswalk,
        )

    with pytest.raises(ImmutableArtifactError, match="artifact_kind mismatch"):
        build_label_opening_receipt_fixture(
            preregistration_receipt=preregistration,
            prediction_seal_receipt=preregistration,
            canonical_crosswalks=(crosswalk,),
            unadmitted_label_bundle_fixture=(
                FixtureOpaqueUnadmittedLabelBundleIdentity.create(
                    bundle_id="unadmitted.labels.v5",
                    bundle_sha256=_hash(60),
                    candidate_inventory_sha256=prediction[
                        "cohort_candidate_inventory"
                    ]["inventory_sha256"],
                    expected_row_inventory_sha256=prediction[
                        "expected_row_inventory"
                    ]["inventory_sha256"],
                )
            ),
            schemas=SCHEMAS,
        )


def test_scientific_input_order_is_exact_in_schema_and_runtime() -> None:
    preregistration, _, _, _ = _chain()
    reordered = copy.deepcopy(preregistration)
    reordered["evaluation_boundary"]["scientific_inputs"].reverse()
    reordered = _reseal(reordered)
    with pytest.raises(ValidationError):
        _validator("preregistration").validate(reordered)
    with pytest.raises(ImmutableArtifactError, match="exact offline boundary"):
        verify_preregistration_receipt_fixture_model(reordered, schemas=SCHEMAS)


def test_external_schema_pins_reject_resealed_schema_hash_aliases() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()
    cases = (
        (
            "preregistration",
            preregistration,
            lambda model: verify_preregistration_receipt_fixture_model(
                model, schemas=SCHEMAS
            ),
        ),
        (
            "prediction",
            prediction,
            lambda model: _verify_prediction(model, preregistration, crosswalk),
        ),
        (
            "label_opening",
            label_opening,
            lambda model: _verify_label(
                model, preregistration, prediction, crosswalk
            ),
        ),
    )
    for schema_name, valid, runtime_verify in cases:
        hostile = copy.deepcopy(valid)
        hostile["schema_sha256"] = "f" * 64
        hostile = _reseal(hostile)
        _validator(schema_name).validate(hostile)
        with pytest.raises(ImmutableArtifactError, match="schema SHA-256 mismatch"):
            runtime_verify(hostile)


def test_schema_runtime_parity_for_valid_and_semantically_hostile_models() -> None:
    preregistration, prediction, label_opening, crosswalk = _chain()
    valid = (
        (
            "preregistration",
            preregistration,
            lambda model: verify_preregistration_receipt_fixture_model(
                model, schemas=SCHEMAS
            ),
        ),
        (
            "prediction",
            prediction,
            lambda model: _verify_prediction(model, preregistration, crosswalk),
        ),
        (
            "label_opening",
            label_opening,
            lambda model: _verify_label(
                model, preregistration, prediction, crosswalk
            ),
        ),
    )
    for schema_name, model, runtime_verify in valid:
        _validator(schema_name).validate(model)
        runtime_verify(model)

    # JSON Schema checks the closed shape; the pure runtime additionally
    # recomputes canonical hashes and exact external predecessor identities.
    semantic_mismatch = copy.deepcopy(prediction)
    semantic_mismatch["canonical_crosswalk_inventory"]["inventory_sha256"] = _hash(
        700
    )
    semantic_mismatch = _reseal(semantic_mismatch)
    _validator("prediction").validate(semantic_mismatch)
    with pytest.raises(ImmutableArtifactError, match="inventory SHA-256 mismatch"):
        _verify_prediction(semantic_mismatch, preregistration, crosswalk)
