from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError

import eval.contracts as exported_contracts
from eval.contracts._common import AuthorityBlockerCode, EvaluatorAuthorityUnavailable
from eval.contracts.launch_crosswalk import (
    CROSSWALK_V2_SCHEMA_ID,
    CROSSWALK_V2_SCHEMA_SHA256,
    CURRENT_M1_WORKLOAD_EXECUTION_AUTHORITIES,
    PRODUCTION_NAMESPACE_ADAPTER_AUTHORITIES,
    PRODUCTION_PRELABEL_TRUST_ANCHORS,
    PRODUCTION_WORKLOAD_EXECUTION_ADMISSION_AUTHORITIES,
    WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID,
    WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256,
    CompleteLaunchKey,
    FixtureV2ExpectedLaunch,
    FixtureV2LaunchObservation,
    FixtureV2ParserAuthority,
    LaunchNamespace,
    V2FixtureAdapterEvidence,
    V2FixtureRetainedInput,
    V2FixtureSourceInput,
    create_launch_inventory_v2_fixture,
    create_launch_source_payload_v2_fixture,
    create_production_launch_inventory,
    create_production_launch_source_payload,
    create_workload_execution_authority,
    create_workload_execution_authority_fixture,
    evaluate_launch_crosswalk_v2,
    evaluate_launch_crosswalk_v2_fixture,
    evaluate_production_launch_crosswalk,
    load_launch_crosswalk_v2,
    load_launch_crosswalk_v2_fixture,
    load_production_launch_crosswalk,
    load_workload_execution_authority,
    require_current_m1_workload_execution_authority,
    verify_launch_crosswalk_v2_fixture_model,
    verify_launch_crosswalk_v2_model,
    verify_launch_inventory_v2_fixture_model,
    verify_production_launch_crosswalk_model,
    verify_production_launch_inventory_model,
    verify_workload_execution_authority_fixture_model,
    verify_workload_execution_authority_model,
)
from eval.contracts.model_seal import RetainedPayloadSnapshot
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    canonical_json_bytes,
    read_regular_snapshot,
    seal_model,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_SCHEMA_PATH = ROOT / "eval/schemas/workload_execution_authority.v1.schema.json"
CROSSWALK_SCHEMA_PATH = ROOT / "eval/schemas/launch_crosswalk.v2.schema.json"
AUTHORITY_SCHEMA_BYTES, AUTHORITY_SCHEMA_SNAPSHOT = RetainedPayloadSnapshot.read_path(
    AUTHORITY_SCHEMA_PATH
)
CROSSWALK_SCHEMA_BYTES, CROSSWALK_SCHEMA_SNAPSHOT = RetainedPayloadSnapshot.read_path(
    CROSSWALK_SCHEMA_PATH
)


def _schema_args() -> dict[str, object]:
    return {
        "schema_bytes": CROSSWALK_SCHEMA_BYTES,
        "schema_snapshot": CROSSWALK_SCHEMA_SNAPSHOT,
        "authority_schema_bytes": AUTHORITY_SCHEMA_BYTES,
        "authority_schema_snapshot": AUTHORITY_SCHEMA_SNAPSHOT,
    }


def _authority_args() -> dict[str, object]:
    return {
        "schema_bytes": AUTHORITY_SCHEMA_BYTES,
        "schema_snapshot": AUTHORITY_SCHEMA_SNAPSHOT,
    }


def _parser_payload(namespace: LaunchNamespace) -> bytes:
    return f"fixture identity parser implementation:{namespace.value}\n".encode()


def _parsers() -> dict[LaunchNamespace, FixtureV2ParserAuthority]:
    return {
        namespace: FixtureV2ParserAuthority(
            namespace,
            f"FIXTURE_{namespace.value}_IDENTITY_PARSER_V2",
            sha256_bytes(_parser_payload(namespace)),
        )
        for namespace in LaunchNamespace
    }


def _launches() -> tuple[FixtureV2ExpectedLaunch, ...]:
    return (
        FixtureV2ExpectedLaunch(0, "_Z5firstv", (8, 1, 1), (128, 1, 1), 0, 3, "sm_86"),
        FixtureV2ExpectedLaunch(1, "_Z6secondv", (4, 2, 1), (64, 2, 1), 0, 7, "sm_86"),
    )


def _authority(nonce: str = "execution-0001", *, metric_like_arg: bool = False) -> dict[str, Any]:
    argv = ("/fixture/vector", "gpc__cycles_elapsed.max") if metric_like_arg else ("/fixture/vector", "--n", "1024")
    return create_workload_execution_authority_fixture(
        executable_sha256="a" * 64,
        argv=argv,
        shape=("n=1024", "dtype=f32"),
        binary_architecture="sm_86",
        gpu_sku="rtx_3090_ti",
        kernel_family="vector",
        expected_launches=_launches(),
        execution_nonce=nonce,
        parser_authorities=_parsers(),
        **_authority_args(),
    )


def _fixture_input(tmp_path: Path, name: str, payload: bytes) -> V2FixtureRetainedInput:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    pinned = read_regular_snapshot(path)
    snapshot = RetainedPayloadSnapshot.from_fixture_payload(pinned.payload, pinned.metadata)
    return V2FixtureRetainedInput(pinned.payload, snapshot)


def _observations(authority: dict[str, Any], namespace: LaunchNamespace) -> tuple[FixtureV2LaunchObservation, ...]:
    context = authority["source_contracts"][namespace.value.lower()]["execution_context_id"]
    rows = []
    for index, expected in enumerate(authority["expected_launch_roster"]):
        key = CompleteLaunchKey.from_model(expected["launch_key"])
        rows.append(
            FixtureV2LaunchObservation.create(
                launch_key=key,
                kernel_family="vector",
                native_context_id=context,
                native_launch_id=f"native-{namespace.value.lower()}-{index}",
                native_stream_id=key.stream_id,
                native_launch_index=index if namespace is LaunchNamespace.NVBIT else None,
                authority_model_seal_sha256=authority["model_seal_sha256"],
            )
        )
    return tuple(rows)


def _adapter(
    tmp_path: Path,
    authority: dict[str, Any],
    namespace: LaunchNamespace,
    *,
    evidence_payload: bytes | None = None,
    token: str = "base",
) -> V2FixtureAdapterEvidence:
    if evidence_payload is None:
        evidence_payload = (
            canonical_json_bytes(authority)
            if namespace is LaunchNamespace.CANONICAL
            else canonical_json_bytes({"fixture_namespace": namespace.value, "complete": True})
        )
    return V2FixtureAdapterEvidence(
        underlying_evidence=_fixture_input(
            tmp_path, f"{token}/{namespace.value}.evidence", evidence_payload
        ),
        parser_implementation=_fixture_input(
            tmp_path, f"{token}/{namespace.value}.parser", _parser_payload(namespace)
        ),
    )


@dataclass(slots=True)
class Case:
    authority: dict[str, Any]
    sources: dict[LaunchNamespace, V2FixtureSourceInput]
    inventories: dict[LaunchNamespace, dict[str, Any]]
    crosswalk: dict[str, Any]


def _case(
    tmp_path: Path,
    *,
    evidence_overrides: dict[LaunchNamespace, bytes] | None = None,
) -> Case:
    authority = _authority()
    sources: dict[LaunchNamespace, V2FixtureSourceInput] = {}
    inventories: dict[LaunchNamespace, dict[str, Any]] = {}
    for namespace in LaunchNamespace:
        adapter = _adapter(
            tmp_path,
            authority,
            namespace,
            evidence_payload=(evidence_overrides or {}).get(namespace),
        )
        payload = create_launch_source_payload_v2_fixture(
            namespace=namespace,
            workload_execution_authority=authority,
            observations=_observations(authority, namespace),
            adapter_evidence=adapter,
            **_schema_args(),
        )
        sources[namespace] = V2FixtureSourceInput(
            _fixture_input(tmp_path, f"base/{namespace.value}.source", payload), adapter
        )
        inventories[namespace] = create_launch_inventory_v2_fixture(
            source_input=sources[namespace],
            workload_execution_authority=authority,
            **_schema_args(),
        )
    crosswalk = evaluate_launch_crosswalk_v2_fixture(
        workload_execution_authority=authority,
        inventories=inventories,
        inventory_sources=sources,
        **_schema_args(),
    )
    return Case(authority, sources, inventories, crosswalk)


def _validator(payload: bytes) -> Draft202012Validator:
    schema = json.loads(payload)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _reseal(model: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(model)
    changed.pop("model_seal_sha256", None)
    return seal_model(changed)


def test_schema_pins_and_fixture_models_validate(tmp_path: Path) -> None:
    assert sha256_bytes(AUTHORITY_SCHEMA_BYTES) == WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256
    assert json.loads(AUTHORITY_SCHEMA_BYTES)["$id"] == WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_ID
    assert sha256_bytes(CROSSWALK_SCHEMA_BYTES) == CROSSWALK_V2_SCHEMA_SHA256
    assert json.loads(CROSSWALK_SCHEMA_BYTES)["$id"] == CROSSWALK_V2_SCHEMA_ID
    case = _case(tmp_path)
    _validator(AUTHORITY_SCHEMA_BYTES).validate(case.authority)
    crosswalk_validator = _validator(CROSSWALK_SCHEMA_BYTES)
    for namespace in LaunchNamespace:
        crosswalk_validator.validate(json.loads(case.sources[namespace].source_projection.payload))
        crosswalk_validator.validate(case.inventories[namespace])
    crosswalk_validator.validate(case.crosswalk)


def test_fixture_ids_are_deterministic_but_not_externally_admitted() -> None:
    first = _authority("execution-0001")
    second = _authority("execution-0002")
    assert first["workload_id"] == second["workload_id"]
    assert first["matched_execution_id"] != second["matched_execution_id"]
    assert first["contract_mode"] == "FIXTURE_ONLY_V2_NON_AUTHORITATIVE"
    assert first["external_admission"] == {
        "status": "FIXTURE_ONLY_NON_AUTHORITATIVE",
        "detached_admission_sha256": None,
        "trust_anchor_id": None,
        "trust_anchor_sha256": None,
    }
    assert first["issuance_claim"]["phase"] == "BEFORE_LABEL_COLLECTION"
    verify_workload_execution_authority_fixture_model(first, **_authority_args())


def test_fixture_four_way_bijection_remains_unavailable(tmp_path: Path) -> None:
    case = _case(tmp_path)
    assert case.crosswalk["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["FIXTURE_ONLY_V2_CROSSWALK"],
    }
    assert case.crosswalk["launch_count"] == 2
    assert all(
        set(row["members"]) == {"canonical", "ncu", "nvbit", "simulator"}
        for row in case.crosswalk["bijection"]
    )
    verify_launch_crosswalk_v2_fixture_model(
        case.crosswalk,
        workload_execution_authority=case.authority,
        inventories=case.inventories,
        inventory_sources=case.sources,
        **_schema_args(),
    )
    assert load_launch_crosswalk_v2_fixture(
        canonical_json_bytes(case.crosswalk),
        workload_execution_authority=case.authority,
        inventories=case.inventories,
        inventory_sources=case.sources,
        **_schema_args(),
    ) == case.crosswalk


def test_every_public_production_v2_path_is_authority_blocked() -> None:
    assert dict(CURRENT_M1_WORKLOAD_EXECUTION_AUTHORITIES) == {}
    assert dict(PRODUCTION_WORKLOAD_EXECUTION_ADMISSION_AUTHORITIES) == {}
    assert dict(PRODUCTION_PRELABEL_TRUST_ANCHORS) == {}
    assert dict(PRODUCTION_NAMESPACE_ADAPTER_AUTHORITIES) == {}
    calls = (
        lambda: create_workload_execution_authority(),
        lambda: verify_workload_execution_authority_model({}),
        lambda: load_workload_execution_authority(b""),
        lambda: create_production_launch_source_payload(),
        lambda: create_production_launch_inventory(),
        lambda: verify_production_launch_inventory_model({}),
        lambda: evaluate_production_launch_crosswalk(),
        lambda: verify_production_launch_crosswalk_model({}),
        lambda: load_production_launch_crosswalk(b""),
        lambda: evaluate_launch_crosswalk_v2(),
        lambda: verify_launch_crosswalk_v2_model({}),
        lambda: load_launch_crosswalk_v2(b""),
    )
    for call in calls:
        with pytest.raises(EvaluatorAuthorityUnavailable) as captured:
            call()
        assert captured.value.code is AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE
    with pytest.raises(EvaluatorAuthorityUnavailable) as captured:
        require_current_m1_workload_execution_authority("sm86-m1")
    assert captured.value.code is AuthorityBlockerCode.COHORT_AUTHORITY_UNAVAILABLE


def test_adapter_binds_exact_evidence_and_parser_bytes(tmp_path: Path) -> None:
    case = _case(tmp_path)
    namespace = LaunchNamespace.NCU
    original = case.sources[namespace]
    changed_evidence = _fixture_input(tmp_path, "changed/evidence", b"different evidence")
    changed_adapter = V2FixtureAdapterEvidence(
        changed_evidence, original.adapter_evidence.parser_implementation
    )
    replay = V2FixtureSourceInput(original.source_projection, changed_adapter)
    with pytest.raises(ImmutableArtifactError, match="adapter evidence binding mismatch"):
        verify_launch_inventory_v2_fixture_model(
            case.inventories[namespace],
            source_input=replay,
            workload_execution_authority=case.authority,
            **_schema_args(),
        )

    missing_binding = json.loads(original.source_projection.payload)
    missing_binding.pop("adapter_evidence")
    with pytest.raises(ValidationError):
        _validator(CROSSWALK_SCHEMA_BYTES).validate(missing_binding)

    bad_parser = V2FixtureAdapterEvidence(
        original.adapter_evidence.underlying_evidence,
        _fixture_input(tmp_path, "changed/parser", b"different parser"),
    )
    with pytest.raises(ImmutableArtifactError, match="parser implementation"):
        create_launch_source_payload_v2_fixture(
            namespace=namespace,
            workload_execution_authority=case.authority,
            observations=_observations(case.authority, namespace),
            adapter_evidence=bad_parser,
            **_schema_args(),
        )


def test_self_attested_production_admission_is_rejected() -> None:
    fixture = _authority()
    hostile = copy.deepcopy(fixture)
    hostile["contract_mode"] = "TARGET_FREE_PRODUCTION"
    hostile["external_admission"] = {
        "status": "EXTERNALLY_ADMITTED",
        "detached_admission_sha256": "b" * 64,
        "trust_anchor_id": "caller-self-attested",
        "trust_anchor_sha256": "c" * 64,
    }
    hostile = _reseal(hostile)
    _validator(AUTHORITY_SCHEMA_BYTES).validate(hostile)
    with pytest.raises(ImmutableArtifactError, match="mode is invalid"):
        verify_workload_execution_authority_fixture_model(hostile, **_authority_args())
    with pytest.raises(EvaluatorAuthorityUnavailable):
        verify_workload_execution_authority_model(hostile)

    missing_anchor = copy.deepcopy(hostile)
    missing_anchor["external_admission"]["trust_anchor_sha256"] = None
    with pytest.raises(ValidationError):
        _validator(AUTHORITY_SCHEMA_BYTES).validate(missing_anchor)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "reordered"])
def test_fixture_source_rejects_incomplete_or_reordered_roster(
    tmp_path: Path, mode: str
) -> None:
    authority = _authority()
    namespace = LaunchNamespace.NVBIT
    rows = _observations(authority, namespace)
    changed = rows[:-1] if mode == "missing" else (rows[0], rows[0]) if mode == "duplicate" else tuple(reversed(rows))
    with pytest.raises(ImmutableArtifactError, match="cardinality|missing, duplicated, or reordered"):
        create_launch_source_payload_v2_fixture(
            namespace=namespace,
            workload_execution_authority=authority,
            observations=changed,
            adapter_evidence=_adapter(tmp_path, authority, namespace),
            **_schema_args(),
        )


def test_uint32_dimensions_are_enforced_by_runtime_and_schema() -> None:
    with pytest.raises(ImmutableArtifactError, match="uint32"):
        FixtureV2ExpectedLaunch(0, "_Z1kv", (1 << 32, 1, 1), (32, 1, 1), 0, 0, "sm_86")
    hostile = copy.deepcopy(_authority())
    hostile["expected_launch_roster"][0]["launch_key"]["grid"]["x"] = 1 << 32
    hostile = _reseal(hostile)
    with pytest.raises(ValidationError):
        _validator(AUTHORITY_SCHEMA_BYTES).validate(hostile)
    with pytest.raises(ImmutableArtifactError, match="uint32"):
        verify_workload_execution_authority_fixture_model(hostile, **_authority_args())


def test_context_stream_and_sorted_nvbit_aliases_are_rejected(tmp_path: Path) -> None:
    authority = _authority()
    key = CompleteLaunchKey.from_model(authority["expected_launch_roster"][0]["launch_key"])
    ncu_context = authority["source_contracts"]["ncu"]["execution_context_id"]
    with pytest.raises(ImmutableArtifactError, match="native stream"):
        FixtureV2LaunchObservation.create(
            launch_key=key,
            kernel_family="vector",
            native_context_id=ncu_context,
            native_launch_id="ncu-0",
            native_stream_id=key.stream_id + 1,
            authority_model_seal_sha256=authority["model_seal_sha256"],
        )

    namespace = LaunchNamespace.NVBIT
    adapter = _adapter(tmp_path, authority, namespace)
    source = json.loads(
        create_launch_source_payload_v2_fixture(
            namespace=namespace,
            workload_execution_authority=authority,
            observations=_observations(authority, namespace),
            adapter_evidence=adapter,
            **_schema_args(),
        )
    )
    source["records"][0]["process_ordinal_binding"]["binding_kind"] = (
        "SORTED_NVBIT_LAUNCH_INDEX_AS_PROCESS_ORDINAL"
    )
    hostile = V2FixtureSourceInput(
        _fixture_input(tmp_path, "sorted/source", canonical_json_bytes(_reseal(source))),
        adapter,
    )
    with pytest.raises(ImmutableArtifactError, match="lacks explicit"):
        create_launch_inventory_v2_fixture(
            source_input=hostile,
            workload_execution_authority=authority,
            **_schema_args(),
        )


def test_binary_architecture_context_and_native_identity_are_exact(tmp_path: Path) -> None:
    case = _case(tmp_path)
    namespace = LaunchNamespace.NCU
    source_model = json.loads(case.sources[namespace].source_projection.payload)
    source_model["executable_sha256"] = "b" * 64
    source_model["architecture"]["binary_architecture"] = "sm_120"
    hostile_source = V2FixtureSourceInput(
        _fixture_input(
            tmp_path, "identity-drift/source", canonical_json_bytes(_reseal(source_model))
        ),
        case.sources[namespace].adapter_evidence,
    )
    with pytest.raises(ImmutableArtifactError, match="authority context mismatch"):
        create_launch_inventory_v2_fixture(
            source_input=hostile_source,
            workload_execution_authority=case.authority,
            **_schema_args(),
        )

    observations = list(_observations(case.authority, namespace))
    second = observations[1]
    observations[1] = FixtureV2LaunchObservation.create(
        launch_key=second.launch_key,
        kernel_family=second.kernel_family,
        native_context_id=observations[0].native_context_id,
        native_launch_id=observations[0].native_launch_id,
        native_stream_id=second.native_stream_id,
        authority_model_seal_sha256=case.authority["model_seal_sha256"],
    )
    with pytest.raises(ImmutableArtifactError, match="duplicate or context-aliased"):
        create_launch_source_payload_v2_fixture(
            namespace=namespace,
            workload_execution_authority=case.authority,
            observations=tuple(observations),
            adapter_evidence=case.sources[namespace].adapter_evidence,
            **_schema_args(),
        )


def test_cross_namespace_underlying_evidence_alias_is_rejected(tmp_path: Path) -> None:
    shared = b"same underlying evidence bytes"
    with pytest.raises(ImmutableArtifactError, match="underlying evidence payloads alias"):
        _case(
            tmp_path,
            evidence_overrides={LaunchNamespace.NCU: shared, LaunchNamespace.NVBIT: shared},
        )


def test_target_guard_is_structural_not_string_heuristic() -> None:
    metric_like = _authority(metric_like_arg=True)
    assert metric_like["argv"][1] == "gpc__cycles_elapsed.max"
    hostile = copy.deepcopy(metric_like)
    hostile["metric_value"] = 42
    hostile = _reseal(hostile)
    with pytest.raises(ValidationError):
        _validator(AUTHORITY_SCHEMA_BYTES).validate(hostile)
    with pytest.raises(ImmutableArtifactError, match="fields are not exact"):
        verify_workload_execution_authority_fixture_model(hostile, **_authority_args())


def test_additive_v2_symbols_are_exported() -> None:
    expected = {
        "create_workload_execution_authority",
        "create_workload_execution_authority_fixture",
        "evaluate_launch_crosswalk_v2",
        "evaluate_launch_crosswalk_v2_fixture",
        "V2FixtureSourceInput",
        "WORKLOAD_EXECUTION_AUTHORITY_SCHEMA_SHA256",
        "CROSSWALK_V2_SCHEMA_SHA256",
    }
    assert expected.issubset(set(exported_contracts.__all__))
    assert exported_contracts.evaluate_launch_crosswalk_v2 is evaluate_launch_crosswalk_v2
