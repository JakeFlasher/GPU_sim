from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import pytest
from jsonschema import Draft202012Validator, ValidationError

from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
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
from eval.contracts.model_seal import RetainedPayloadSnapshot, canonical_model_bytes
from eval.contracts.state_lifetime import (
    PRODUCTION_STATE_EVIDENCE_AUTHORITIES,
    SCHEMA_ID,
    SCHEMA_SHA256,
    CacheCondition,
    DeviceScope,
    EvidenceBasis,
    ExecutionRelation,
    IsolationState,
    LaunchScope,
    ProcessScope,
    StateLifetimeEvidence,
    StateOrigin,
    StreamScope,
    classify_state_lifetime,
    classify_state_lifetime_fixture,
    create_state_evidence_payload,
    create_state_evidence_payload_fixture,
    load_state_lifetime_classification,
    load_state_lifetime_fixture_classification,
    verify_state_evidence_fixture_payload,
    verify_state_evidence_payload,
    verify_state_lifetime_fixture_model,
    verify_state_lifetime_model,
)
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    read_regular_snapshot,
    seal_model,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
STATE_SCHEMA_PATH = ROOT / "eval" / "schemas" / "state_lifetime.v1.schema.json"
CROSSWALK_SCHEMA_PATH = ROOT / "eval" / "schemas" / "launch_crosswalk.v1.schema.json"
STATE_SCHEMA_BYTES, STATE_SCHEMA_SNAPSHOT = RetainedPayloadSnapshot.read_path(
    STATE_SCHEMA_PATH
)
CROSSWALK_SCHEMA_BYTES, CROSSWALK_SCHEMA_SNAPSHOT = (
    RetainedPayloadSnapshot.read_path(CROSSWALK_SCHEMA_PATH)
)
SIGNATURE = LaunchSignature("kernel_main", (8, 2, 1), (128, 1, 1))
_SOURCE_KINDS = {
    LaunchNamespace.CANONICAL: "CANONICAL_LAUNCH_SOURCE",
    LaunchNamespace.NCU: "NCU_LAUNCH_SOURCE",
    LaunchNamespace.NVBIT: "NVBIT_LAUNCH_SOURCE",
    LaunchNamespace.SIMULATOR: "SIMULATOR_LAUNCH_SOURCE",
}


def _validator() -> Draft202012Validator:
    schema = json.loads(STATE_SCHEMA_BYTES)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _retained_file(path: Path, payload: bytes) -> tuple[bytes, RetainedPayloadSnapshot]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    pinned = read_regular_snapshot(path)
    return pinned.payload, RetainedPayloadSnapshot.from_fixture_payload(
        pinned.payload, pinned.metadata
    )


def _fixture_crosswalk(tmp_path: Path, *, token: str = "base") -> dict[str, Any]:
    sources: dict[LaunchNamespace, SourceArtifactDescriptor] = {}
    payloads: dict[LaunchNamespace, bytes] = {}
    records: dict[LaunchNamespace, tuple[LaunchRecord, ...]] = {}
    inventories: dict[LaunchNamespace, dict[str, Any]] = {}
    for namespace in LaunchNamespace:
        role = "TARGET" if namespace is LaunchNamespace.CANONICAL else "COLD"
        payload, snapshot = _retained_file(
            tmp_path / token / f"{namespace.value.lower()}.launches",
            fixture_launch_source_payload(
                namespace,
                (FixtureLaunchSpec(SIGNATURE, 0, role),),
            ),
        )
        source = SourceArtifactDescriptor.from_fixture_payload(
            namespace=namespace,
            artifact_kind=_SOURCE_KINDS[namespace],
            payload=payload,
            snapshot=snapshot,
        )
        sources[namespace] = source
        payloads[namespace] = payload
        records[namespace] = (
            LaunchRecord.create(
                source=source,
                signature=SIGNATURE,
                launch_ordinal=0,
                launch_role=role,
            ),
        )
        inventories[namespace] = create_launch_inventory_fixture(
            source=source,
            source_payload=payload,
            candidates=records[namespace],
            schema_bytes=CROSSWALK_SCHEMA_BYTES,
            schema_snapshot=CROSSWALK_SCHEMA_SNAPSHOT,
        )
    bindings = {
        namespace: IndependentLaunchBinding.create_fixture(
            inventory=inventories[namespace],
            canonical_launch=records[LaunchNamespace.CANONICAL][0],
            selected_launch=records[namespace][0],
            inventory_schema_bytes=CROSSWALK_SCHEMA_BYTES,
            inventory_schema_snapshot=CROSSWALK_SCHEMA_SNAPSHOT,
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
        schema_bytes=CROSSWALK_SCHEMA_BYTES,
        schema_snapshot=CROSSWALK_SCHEMA_SNAPSHOT,
        inventory_schema_bytes=CROSSWALK_SCHEMA_BYTES,
        inventory_schema_snapshot=CROSSWALK_SCHEMA_SNAPSHOT,
    )


def _record(crosswalk: dict[str, Any], **overrides: Any) -> StateLifetimeEvidence:
    arguments: dict[str, Any] = {
        "evidence_record_id": "state-record-1",
        "evidence_basis": EvidenceBasis.EXACT,
        "state_origin": StateOrigin.FRESH,
        "cache_condition": CacheCondition.COLD,
        "launch_scope": LaunchScope.SINGLE_LAUNCH,
        "execution_relation": ExecutionRelation.INDEPENDENT,
        "isolation_state": IsolationState.ISOLATED,
        "stream_scope": StreamScope.SINGLE_STREAM,
        "process_scope": ProcessScope.SINGLE_PROCESS,
        "device_scope": DeviceScope.SINGLE_DEVICE,
        "crosswalk_model": crosswalk,
        "crosswalk_schema_bytes": CROSSWALK_SCHEMA_BYTES,
        "crosswalk_schema_snapshot": CROSSWALK_SCHEMA_SNAPSHOT,
        "inventory_schema_bytes": CROSSWALK_SCHEMA_BYTES,
        "inventory_schema_snapshot": CROSSWALK_SCHEMA_SNAPSHOT,
    }
    arguments.update(overrides)
    return StateLifetimeEvidence.create_fixture(**arguments)


def _payload(
    tmp_path: Path,
    records: Iterable[StateLifetimeEvidence],
    *,
    name: str = "state-evidence.json",
) -> tuple[dict[str, Any], bytes, RetainedPayloadSnapshot]:
    model = create_state_evidence_payload_fixture(
        records,
        schema_bytes=STATE_SCHEMA_BYTES,
        schema_snapshot=STATE_SCHEMA_SNAPSHOT,
    )
    payload = canonical_model_bytes(model)
    retained, snapshot = _retained_file(tmp_path / name, payload)
    return model, retained, snapshot


def _snapshot_payload(
    tmp_path: Path, payload: bytes, *, name: str
) -> RetainedPayloadSnapshot:
    retained, snapshot = _retained_file(tmp_path / name, payload)
    assert retained == payload
    return snapshot


def _classification_arguments(
    *,
    evidence_payload: bytes | None,
    evidence_snapshot: RetainedPayloadSnapshot | None,
    evidence_record_id: str,
    crosswalk: dict[str, Any],
    evidence_schema_bytes: bytes = STATE_SCHEMA_BYTES,
    evidence_schema_snapshot: RetainedPayloadSnapshot = STATE_SCHEMA_SNAPSHOT,
) -> dict[str, Any]:
    return {
        "evidence_payload": evidence_payload,
        "evidence_snapshot": evidence_snapshot,
        "evidence_record_id": evidence_record_id,
        "crosswalk_model": crosswalk,
        "schema_bytes": STATE_SCHEMA_BYTES,
        "schema_snapshot": STATE_SCHEMA_SNAPSHOT,
        "evidence_schema_bytes": evidence_schema_bytes,
        "evidence_schema_snapshot": evidence_schema_snapshot,
        "crosswalk_schema_bytes": CROSSWALK_SCHEMA_BYTES,
        "crosswalk_schema_snapshot": CROSSWALK_SCHEMA_SNAPSHOT,
        "inventory_schema_bytes": CROSSWALK_SCHEMA_BYTES,
        "inventory_schema_snapshot": CROSSWALK_SCHEMA_SNAPSHOT,
    }


def _classify(
    *,
    evidence_payload: bytes | None,
    evidence_snapshot: RetainedPayloadSnapshot | None,
    evidence_record_id: str,
    crosswalk: dict[str, Any],
    evidence_schema_bytes: bytes = STATE_SCHEMA_BYTES,
    evidence_schema_snapshot: RetainedPayloadSnapshot = STATE_SCHEMA_SNAPSHOT,
) -> dict[str, Any]:
    return classify_state_lifetime_fixture(
        **_classification_arguments(
            evidence_payload=evidence_payload,
            evidence_snapshot=evidence_snapshot,
            evidence_record_id=evidence_record_id,
            crosswalk=crosswalk,
            evidence_schema_bytes=evidence_schema_bytes,
            evidence_schema_snapshot=evidence_schema_snapshot,
        )
    )


def _verify(model: dict[str, Any], **inputs: Any) -> None:
    verify_state_lifetime_fixture_model(
        model, **_classification_arguments(**inputs)
    )


def _reseal(model: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(model)
    changed.pop("model_seal_sha256", None)
    return seal_model(changed)


def test_state_schema_bytes_are_runtime_pinned() -> None:
    assert sha256_bytes(STATE_SCHEMA_BYTES) == SCHEMA_SHA256
    assert json.loads(STATE_SCHEMA_BYTES)["$id"] == SCHEMA_ID


def test_production_state_authority_registry_is_empty_and_all_paths_block(
    tmp_path: Path,
) -> None:
    assert dict(PRODUCTION_STATE_EVIDENCE_AUTHORITIES) == {}
    calls = (
        lambda: StateLifetimeEvidence.create(),
        lambda: create_state_evidence_payload(
            (),
            schema_bytes=STATE_SCHEMA_BYTES,
            schema_snapshot=STATE_SCHEMA_SNAPSHOT,
        ),
        lambda: verify_state_evidence_payload(
            {},
            schema_bytes=STATE_SCHEMA_BYTES,
            schema_snapshot=STATE_SCHEMA_SNAPSHOT,
        ),
        lambda: classify_state_lifetime(),
        lambda: verify_state_lifetime_model({}),
        lambda: load_state_lifetime_classification(b"{}"),
    )
    for call in calls:
        with pytest.raises(EvaluatorAuthorityUnavailable) as captured:
            call()
        assert (
            captured.value.code
            is AuthorityBlockerCode.STATE_EVIDENCE_AUTHORITY_UNAVAILABLE
        )

    _, fixture_snapshot = _retained_file(
        tmp_path / "fabricated-schema-fixture.json", STATE_SCHEMA_BYTES
    )
    with pytest.raises(EvaluatorAuthorityUnavailable) as acquisition_blocker:
        fixture_snapshot.require_production_acquisition("state evidence")
    assert (
        acquisition_blocker.value.code
        is AuthorityBlockerCode.RETAINED_DESCRIPTOR_ACQUISITION_REQUIRED
    )


def test_exact_retained_fixture_evidence_is_production_unavailable(
    tmp_path: Path,
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    payload_model, payload, snapshot = _payload(tmp_path, (evidence,))
    inputs = {
        "evidence_payload": payload,
        "evidence_snapshot": snapshot,
        "evidence_record_id": evidence.evidence_record_id,
        "crosswalk": crosswalk,
    }
    model = _classify(**inputs)

    assert model["contract_mode"] == "FIXTURE_ONLY"
    assert model["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["FIXTURE_ONLY_STATE_EVIDENCE"],
    }
    assert model["evidence_source"]["payload_snapshot"] == snapshot.to_model()
    assert model["evidence_source"]["payload_model_seal_sha256"] == payload_model[
        "model_seal_sha256"
    ]
    assert model["evidence_source"]["crosswalk_model_seal_sha256"] == crosswalk[
        "model_seal_sha256"
    ]
    assert set(model["evidence_source"]["selected_qualified_launch_ids"]) == {
        "canonical",
        "ncu",
        "nvbit",
        "simulator",
    }
    assert verify_state_evidence_fixture_payload(
        payload_model,
        schema_bytes=STATE_SCHEMA_BYTES,
        schema_snapshot=STATE_SCHEMA_SNAPSHOT,
    ) == (evidence,)
    _verify(model, **inputs)
    _validator().validate(payload_model)
    _validator().validate(model)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"cache_condition": CacheCondition.WARM}, "WARM_STATE_REJECTED"),
        ({"state_origin": StateOrigin.PERSISTENT}, "PERSISTENT_STATE_REJECTED"),
        (
            {"execution_relation": ExecutionRelation.CONCURRENT},
            "CONCURRENT_EXECUTION_REJECTED",
        ),
        (
            {"execution_relation": ExecutionRelation.DEPENDENT},
            "DEPENDENT_EXECUTION_REJECTED",
        ),
        (
            {"isolation_state": IsolationState.NON_ISOLATED},
            "NON_ISOLATED_EXECUTION_REJECTED",
        ),
        ({"launch_scope": LaunchScope.MULTI_LAUNCH}, "MULTI_LAUNCH_REJECTED"),
        ({"stream_scope": StreamScope.MULTI_STREAM}, "MULTI_STREAM_REJECTED"),
        ({"process_scope": ProcessScope.MULTI_PROCESS}, "MULTI_PROCESS_REJECTED"),
        ({"device_scope": DeviceScope.MULTI_DEVICE}, "MULTI_DEVICE_REJECTED"),
    ],
)
def test_known_nonindependent_or_nonfresh_cases_are_rejected(
    tmp_path: Path, override: dict[str, Any], reason: str
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk, **override)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )

    assert model["outcome"]["status"] == "REJECTED"
    assert reason in model["outcome"]["reason_codes"]
    _validator().validate(model)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"evidence_basis": EvidenceBasis.INFERRED}, "EXACT_EVIDENCE_UNAVAILABLE"),
        ({"evidence_basis": EvidenceBasis.UNKNOWN}, "EXACT_EVIDENCE_UNAVAILABLE"),
        ({"state_origin": StateOrigin.UNKNOWN}, "FRESH_STATE_UNAVAILABLE"),
        ({"cache_condition": CacheCondition.EVICTED}, "EVICTED_IS_NOT_EXACT_COLD"),
        ({"cache_condition": CacheCondition.UNKNOWN}, "COLD_STATE_UNAVAILABLE"),
        ({"launch_scope": LaunchScope.UNKNOWN}, "SINGLE_LAUNCH_UNAVAILABLE"),
        (
            {"execution_relation": ExecutionRelation.UNKNOWN},
            "INDEPENDENCE_UNAVAILABLE",
        ),
        ({"isolation_state": IsolationState.UNKNOWN}, "ISOLATION_UNAVAILABLE"),
        ({"stream_scope": StreamScope.UNKNOWN}, "SINGLE_STREAM_UNAVAILABLE"),
        ({"process_scope": ProcessScope.UNKNOWN}, "SINGLE_PROCESS_UNAVAILABLE"),
        ({"device_scope": DeviceScope.UNKNOWN}, "SINGLE_DEVICE_UNAVAILABLE"),
    ],
)
def test_ambiguous_or_unproven_state_cases_are_unavailable(
    tmp_path: Path, override: dict[str, Any], reason: str
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk, **override)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )

    assert model["outcome"]["status"] == "UNAVAILABLE"
    assert reason in model["outcome"]["reason_codes"]
    _validator().validate(model)


def test_explicit_rejection_precedes_missing_exactness(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(
        crosswalk,
        evidence_basis=EvidenceBasis.UNKNOWN,
        cache_condition=CacheCondition.WARM,
        execution_relation=ExecutionRelation.DEPENDENT,
        isolation_state=IsolationState.NON_ISOLATED,
    )
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    assert model["outcome"] == {
        "status": "REJECTED",
        "reason_codes": [
            "WARM_STATE_REJECTED",
            "DEPENDENT_EXECUTION_REJECTED",
            "NON_ISOLATED_EXECUTION_REJECTED",
        ],
    }


def test_absent_payload_is_typed_unavailable(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    model = _classify(
        evidence_payload=None,
        evidence_snapshot=None,
        evidence_record_id="state-record-1",
        crosswalk=crosswalk,
    )
    assert model["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["EXACT_EVIDENCE_PAYLOAD_UNAVAILABLE"],
    }
    assert model["evidence"] is None
    assert model["evidence_source"] is None
    _validator().validate(model)


def test_fixture_state_payload_rejects_production_acquisition_class(
    tmp_path: Path,
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    payload_model = create_state_evidence_payload_fixture(
        (evidence,),
        schema_bytes=STATE_SCHEMA_BYTES,
        schema_snapshot=STATE_SCHEMA_SNAPSHOT,
    )
    payload = canonical_model_bytes(payload_model)
    production_path = tmp_path / "production-acquired-state-evidence.json"
    production_path.write_bytes(payload)
    retained, production_snapshot = RetainedPayloadSnapshot.read_path(production_path)
    model = _classify(
        evidence_payload=retained,
        evidence_snapshot=production_snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    assert model["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["EVIDENCE_PAYLOAD_INVALID"],
    }
    _validator().validate(model)


def test_payload_must_match_exact_retained_snapshot(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    _, payload, _ = _payload(tmp_path, (evidence,))
    mismatched_snapshot = _snapshot_payload(
        tmp_path, payload + b" ", name="mismatched-evidence.json"
    )
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=mismatched_snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    assert model["outcome"]["reason_codes"] == ["EVIDENCE_PAYLOAD_INVALID"]


@pytest.mark.parametrize(
    "malformed",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"value":NaN}',
        b'[]',
    ],
)
def test_malformed_or_duplicate_key_payload_is_unavailable(
    tmp_path: Path, malformed: bytes
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    snapshot = _snapshot_payload(tmp_path, malformed, name="malformed-evidence.json")
    model = _classify(
        evidence_payload=malformed,
        evidence_snapshot=snapshot,
        evidence_record_id="state-record-1",
        crosswalk=crosswalk,
    )
    assert model["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["EVIDENCE_PAYLOAD_INVALID"],
    }


def test_wrong_retained_evidence_schema_bytes_are_unavailable(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    changed_schema = STATE_SCHEMA_BYTES + b"\n"
    changed_schema_snapshot = _snapshot_payload(
        tmp_path, changed_schema, name="changed-state-schema.json"
    )
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
        evidence_schema_bytes=changed_schema,
        evidence_schema_snapshot=changed_schema_snapshot,
    )
    assert model["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["EVIDENCE_SCHEMA_UNAVAILABLE"],
    }


def test_payload_header_schema_hash_mismatch_is_unavailable(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    payload_model, _, _ = _payload(tmp_path, (evidence,))
    payload_model["schema_sha256"] = "f" * 64
    payload_model = _reseal(payload_model)
    payload = canonical_model_bytes(payload_model)
    snapshot = _snapshot_payload(tmp_path, payload, name="wrong-payload-schema.json")
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    assert model["outcome"]["reason_codes"] == ["EVIDENCE_PAYLOAD_INVALID"]


def test_missing_and_duplicate_record_membership_are_unavailable(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    other = _record(crosswalk, evidence_record_id="other-record")
    _, missing_payload, missing_snapshot = _payload(tmp_path, (other,), name="missing.json")
    missing = _classify(
        evidence_payload=missing_payload,
        evidence_snapshot=missing_snapshot,
        evidence_record_id="state-record-1",
        crosswalk=crosswalk,
    )
    assert missing["outcome"]["reason_codes"] == ["EVIDENCE_RECORD_UNAVAILABLE"]

    evidence = _record(crosswalk)
    payload_model = create_state_evidence_payload_fixture(
        (evidence,),
        schema_bytes=STATE_SCHEMA_BYTES,
        schema_snapshot=STATE_SCHEMA_SNAPSHOT,
    )
    payload_model["records"].append(copy.deepcopy(payload_model["records"][0]))
    payload_model = _reseal(payload_model)
    duplicate_payload = canonical_model_bytes(payload_model)
    duplicate_snapshot = _snapshot_payload(
        tmp_path, duplicate_payload, name="duplicate-membership.json"
    )
    duplicate = _classify(
        evidence_payload=duplicate_payload,
        evidence_snapshot=duplicate_snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    assert duplicate["outcome"]["status"] == "UNAVAILABLE"
    assert duplicate["outcome"]["reason_codes"] == ["EVIDENCE_PAYLOAD_INVALID"]


def test_evidence_crosswalk_seal_and_selected_ids_are_anti_replay_bound(
    tmp_path: Path,
) -> None:
    original = _fixture_crosswalk(tmp_path, token="original")
    replacement = _fixture_crosswalk(tmp_path, token="replacement")
    evidence = _record(original)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=replacement,
    )
    assert model["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["EVIDENCE_CROSSWALK_BINDING_UNAVAILABLE"],
    }


def test_invalid_fixture_crosswalk_is_unavailable(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    invalid = copy.deepcopy(crosswalk)
    invalid["model_seal_sha256"] = "0" * 64
    model = _classify(
        evidence_payload=None,
        evidence_snapshot=None,
        evidence_record_id="state-record-1",
        crosswalk=invalid,
    )
    assert model["outcome"]["reason_codes"] == ["CROSSWALK_UNAVAILABLE"]


def test_evidence_payload_seal_is_order_independent_and_membership_sorted(
    tmp_path: Path,
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    first = _record(crosswalk, evidence_record_id="a-record")
    second = _record(crosswalk, evidence_record_id="b-record")
    forward = create_state_evidence_payload_fixture(
        (first, second),
        schema_bytes=STATE_SCHEMA_BYTES,
        schema_snapshot=STATE_SCHEMA_SNAPSHOT,
    )
    reverse = create_state_evidence_payload_fixture(
        (second, first),
        schema_bytes=STATE_SCHEMA_BYTES,
        schema_snapshot=STATE_SCHEMA_SNAPSHOT,
    )
    assert forward == reverse
    assert [item["evidence_record_id"] for item in forward["records"]] == [
        "a-record",
        "b-record",
    ]


def test_classification_verifier_recomputes_from_exact_bytes(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    inputs = {
        "evidence_payload": payload,
        "evidence_snapshot": snapshot,
        "evidence_record_id": evidence.evidence_record_id,
        "crosswalk": crosswalk,
    }
    valid = _classify(**inputs)
    changed = copy.deepcopy(valid)
    changed["evidence"]["cache_condition"] = "WARM"
    changed = _reseal(changed)
    with pytest.raises(ImmutableArtifactError, match="does not match retained bytes"):
        _verify(changed, **inputs)


def test_state_loader_rejects_duplicate_json_and_unknown_version(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    inputs = {
        "evidence_payload": payload,
        "evidence_snapshot": snapshot,
        "evidence_record_id": evidence.evidence_record_id,
        "crosswalk": crosswalk,
    }
    model = _classify(**inputs)
    serialized = canonical_model_bytes(model)
    verification = _classification_arguments(**inputs)
    assert load_state_lifetime_fixture_classification(serialized, **verification) == model

    duplicate = serialized[:-1] + b',"schema_version":1}'
    with pytest.raises(ImmutableArtifactError, match="duplicate JSON key"):
        load_state_lifetime_fixture_classification(duplicate, **verification)

    changed = copy.deepcopy(model)
    changed["schema_version"] = 2
    changed = _reseal(changed)
    with pytest.raises(ImmutableArtifactError, match="schema_version mismatch"):
        load_state_lifetime_fixture_classification(
            canonical_model_bytes(changed), **verification
        )


def test_schema_and_runtime_reject_boolean_versions_and_terminal_newlines(
    tmp_path: Path,
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    payload_model, payload, snapshot = _payload(tmp_path, (evidence,))
    validator = _validator()

    bad_version = copy.deepcopy(payload_model)
    bad_version["schema_version"] = True
    bad_version = _reseal(bad_version)
    with pytest.raises(ValidationError):
        validator.validate(bad_version)
    with pytest.raises(ImmutableArtifactError, match="schema_version"):
        verify_state_evidence_fixture_payload(
            bad_version,
            schema_bytes=STATE_SCHEMA_BYTES,
            schema_snapshot=STATE_SCHEMA_SNAPSHOT,
        )

    bad_hash = copy.deepcopy(payload_model)
    bad_hash["records"][0]["selected_qualified_launch_ids"]["ncu"] += "\n"
    bad_hash = _reseal(bad_hash)
    with pytest.raises(ValidationError):
        validator.validate(bad_hash)
    with pytest.raises(ImmutableArtifactError, match="lowercase 64-hex"):
        verify_state_evidence_fixture_payload(
            bad_hash,
            schema_bytes=STATE_SCHEMA_BYTES,
            schema_snapshot=STATE_SCHEMA_SNAPSHOT,
        )

    bad_id = copy.deepcopy(payload_model)
    bad_id["records"][0]["evidence_record_id"] += "\n"
    bad_id = _reseal(bad_id)
    with pytest.raises(ValidationError):
        validator.validate(bad_id)
    with pytest.raises(ImmutableArtifactError, match="forbidden control"):
        verify_state_evidence_fixture_payload(
            bad_id,
            schema_bytes=STATE_SCHEMA_BYTES,
            schema_snapshot=STATE_SCHEMA_SNAPSHOT,
        )

    valid = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    validator.validate(valid)

    bad_classification_version = copy.deepcopy(valid)
    bad_classification_version["schema_version"] = True
    bad_classification_version = _reseal(bad_classification_version)
    with pytest.raises(ValidationError):
        validator.validate(bad_classification_version)
    with pytest.raises(ImmutableArtifactError, match="schema_version"):
        _verify(
            bad_classification_version,
            evidence_payload=payload,
            evidence_snapshot=snapshot,
            evidence_record_id=evidence.evidence_record_id,
            crosswalk=crosswalk,
        )


def test_rejected_and_unavailable_reason_vocabularies_are_disjoint(
    tmp_path: Path,
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    inputs = {
        "evidence_payload": payload,
        "evidence_snapshot": snapshot,
        "evidence_record_id": evidence.evidence_record_id,
        "crosswalk": crosswalk,
    }
    valid = _classify(**inputs)
    for status, reason in (
        ("UNAVAILABLE", "WARM_STATE_REJECTED"),
        ("REJECTED", "EXACT_EVIDENCE_UNAVAILABLE"),
    ):
        invalid = copy.deepcopy(valid)
        invalid["outcome"] = {"status": status, "reason_codes": [reason]}
        invalid = _reseal(invalid)
        with pytest.raises(ValidationError):
            _validator().validate(invalid)
        with pytest.raises(ImmutableArtifactError, match="reason vocabulary"):
            _verify(invalid, **inputs)


def test_state_schema_is_closed_against_metrics_and_unknown_enums(tmp_path: Path) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    valid = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    validator = _validator()

    extra = copy.deepcopy(valid)
    extra["numeric_metric_value"] = 123
    with pytest.raises(ValidationError):
        validator.validate(extra)

    nested = copy.deepcopy(valid)
    nested["evidence"]["metric_values"] = {"cycles": 123}
    with pytest.raises(ValidationError):
        validator.validate(nested)

    unknown = copy.deepcopy(valid)
    unknown["evidence"]["isolation_state"] = "ASSUMED_ISOLATED"
    with pytest.raises(ValidationError):
        validator.validate(unknown)


def test_cycles_remain_pending_unavailable_unscored_and_m5_disabled(
    tmp_path: Path,
) -> None:
    crosswalk = _fixture_crosswalk(tmp_path)
    evidence = _record(crosswalk)
    _, payload, snapshot = _payload(tmp_path, (evidence,))
    model = _classify(
        evidence_payload=payload,
        evidence_snapshot=snapshot,
        evidence_record_id=evidence.evidence_record_id,
        crosswalk=crosswalk,
    )
    boundary = model["evaluation_boundary"]
    assert boundary["m5"] == {
        "status": "M5_NOT_STARTED",
        "collection_status": "M5_COLLECT_DISABLED",
    }
    assert boundary["cycles"] == {
        "metric_id": "gpc__cycles_elapsed.max",
        "implementation_status": "required_unimplemented",
        "evaluation_state": "pending",
        "outcome": "unavailable",
        "score_status": "unscored",
    }
    assert [item["name"] for item in boundary["scientific_inputs"]] == [
        "CYCLE_MAPPING",
        "CYCLE_COMPOSITION",
        "COHORT",
        "PROTOCOL_NUMERIC_SETTINGS",
        "COVERAGE",
        "TRUST",
        "OPERATOR_APPROVAL",
    ]
    assert all(item["status"] == "UNRESOLVED" for item in boundary["scientific_inputs"])
