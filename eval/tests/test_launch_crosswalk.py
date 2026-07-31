from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pytest
from jsonschema import Draft202012Validator, ValidationError

from eval.contracts._common import (
    AuthorityBlockerCode,
    EvaluatorAuthorityUnavailable,
)
from eval.contracts.launch_crosswalk import (
    PRODUCTION_BINDING_AUTHORITIES,
    PRODUCTION_CROSS_NAMESPACE_PROVENANCE_AUTHORITIES,
    PRODUCTION_LAUNCH_PARSER_AUTHORITIES,
    SCHEMA_ID,
    SCHEMA_SHA256,
    FIXTURE_BINDING_AUTHORITY_SHA256,
    FixtureLaunchSpec,
    IndependentLaunchBinding,
    LaunchNamespace,
    LaunchRecord,
    LaunchSignature,
    SourceArtifactDescriptor,
    create_launch_inventory,
    create_launch_inventory_fixture,
    evaluate_launch_crosswalk,
    evaluate_launch_crosswalk_fixture,
    fixture_launch_source_payload,
    load_launch_crosswalk,
    load_launch_crosswalk_fixture,
    verify_launch_crosswalk_fixture_model,
    verify_launch_crosswalk_model,
    verify_launch_inventory_fixture_model,
    verify_launch_inventory_model,
)
from eval.contracts.model_seal import RetainedPayloadSnapshot, canonical_model_bytes
from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    read_regular_snapshot,
    seal_model,
    sha256_bytes,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "eval" / "schemas" / "launch_crosswalk.v1.schema.json"
SCHEMA_BYTES, SCHEMA_SNAPSHOT = RetainedPayloadSnapshot.read_path(SCHEMA_PATH)
SIGNATURE = LaunchSignature(
    kernel_name="kernel_main",
    grid_dim=(8, 2, 1),
    block_dim=(128, 1, 1),
)
OTHER_SIGNATURE = LaunchSignature(
    kernel_name="kernel_other",
    grid_dim=(4, 1, 1),
    block_dim=(64, 1, 1),
)
_SOURCE_KINDS = {
    LaunchNamespace.CANONICAL: "CANONICAL_LAUNCH_SOURCE",
    LaunchNamespace.NCU: "NCU_LAUNCH_SOURCE",
    LaunchNamespace.NVBIT: "NVBIT_LAUNCH_SOURCE",
    LaunchNamespace.SIMULATOR: "SIMULATOR_LAUNCH_SOURCE",
}


@dataclass(slots=True)
class CrosswalkCase:
    sources: dict[LaunchNamespace, SourceArtifactDescriptor]
    source_payloads: dict[LaunchNamespace, bytes]
    records: dict[LaunchNamespace, tuple[LaunchRecord, ...]]
    inventories: dict[LaunchNamespace, dict[str, Any]]

    @property
    def canonical_launch(self) -> LaunchRecord:
        return self.records[LaunchNamespace.CANONICAL][0]


def _schema_arguments() -> dict[str, object]:
    return {
        "schema_bytes": SCHEMA_BYTES,
        "schema_snapshot": SCHEMA_SNAPSHOT,
        "inventory_schema_bytes": SCHEMA_BYTES,
        "inventory_schema_snapshot": SCHEMA_SNAPSHOT,
    }


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_BYTES)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _retained_file(path: Path, payload: bytes) -> tuple[bytes, RetainedPayloadSnapshot]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    pinned = read_regular_snapshot(path)
    return pinned.payload, RetainedPayloadSnapshot.from_fixture_payload(
        pinned.payload, pinned.metadata
    )


def _source(
    tmp_path: Path,
    namespace: LaunchNamespace,
    token: str,
    specs: tuple[tuple[LaunchSignature, int, str], ...],
) -> tuple[SourceArtifactDescriptor, bytes]:
    payload, snapshot = _retained_file(
        tmp_path / token / f"{namespace.value.lower()}.launches",
        fixture_launch_source_payload(
            namespace,
            tuple(
                FixtureLaunchSpec(signature, ordinal, role)
                for signature, ordinal, role in specs
            ),
        ),
    )
    return (
        SourceArtifactDescriptor.from_fixture_payload(
            namespace=namespace,
            artifact_kind=_SOURCE_KINDS[namespace],
            payload=payload,
            snapshot=snapshot,
        ),
        payload,
    )


def _make_case(
    tmp_path: Path,
    *,
    token: str = "base",
    canonical_signature: LaunchSignature = SIGNATURE,
    source_specs: dict[
        LaunchNamespace, tuple[tuple[LaunchSignature, int, str], ...]
    ]
    | None = None,
) -> CrosswalkCase:
    specs = {
        LaunchNamespace.CANONICAL: ((canonical_signature, 0, "TARGET"),),
        LaunchNamespace.NCU: ((canonical_signature, 0, "COLD"),),
        LaunchNamespace.NVBIT: ((canonical_signature, 0, "COLD"),),
        LaunchNamespace.SIMULATOR: ((canonical_signature, 0, "COLD"),),
    }
    if source_specs:
        specs.update(source_specs)
    sources: dict[LaunchNamespace, SourceArtifactDescriptor] = {}
    payloads: dict[LaunchNamespace, bytes] = {}
    for namespace in LaunchNamespace:
        sources[namespace], payloads[namespace] = _source(
            tmp_path, namespace, token, specs[namespace]
        )
    records: dict[LaunchNamespace, tuple[LaunchRecord, ...]] = {
        LaunchNamespace.CANONICAL: (
            LaunchRecord.create(
                source=sources[LaunchNamespace.CANONICAL],
                signature=canonical_signature,
                launch_ordinal=0,
                launch_role="TARGET",
            ),
        )
    }
    for namespace in (
        LaunchNamespace.NCU,
        LaunchNamespace.NVBIT,
        LaunchNamespace.SIMULATOR,
    ):
        records[namespace] = tuple(
            LaunchRecord.create(
                source=sources[namespace],
                signature=signature,
                launch_ordinal=ordinal,
                launch_role=role,
            )
            for signature, ordinal, role in specs[namespace]
        )
    inventories = {
        namespace: create_launch_inventory_fixture(
            source=sources[namespace],
            source_payload=payloads[namespace],
            candidates=records[namespace],
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        )
        for namespace in LaunchNamespace
    }
    return CrosswalkCase(sources, payloads, records, inventories)


def _binding(
    case: CrosswalkCase,
    namespace: LaunchNamespace,
    *,
    selected: LaunchRecord | None = None,
    inventory: dict[str, Any] | None = None,
) -> IndependentLaunchBinding:
    return IndependentLaunchBinding.create_fixture(
        inventory=case.inventories[namespace] if inventory is None else inventory,
        canonical_launch=case.canonical_launch,
        selected_launch=case.records[namespace][0] if selected is None else selected,
        inventory_schema_bytes=SCHEMA_BYTES,
        inventory_schema_snapshot=SCHEMA_SNAPSHOT,
    )


def _evaluate(
    case: CrosswalkCase,
    *,
    inventories: dict[LaunchNamespace, dict[str, Any]] | None = None,
    candidates: dict[LaunchNamespace, Iterable[LaunchRecord]] | None = None,
    bindings: dict[LaunchNamespace, IndependentLaunchBinding | None] | None = None,
) -> dict[str, Any]:
    installed = case.inventories | (inventories or {})
    supplied = case.records | {
        namespace: tuple(values) for namespace, values in (candidates or {}).items()
    }
    selected_bindings: dict[LaunchNamespace, IndependentLaunchBinding | None] = {
        LaunchNamespace.NCU: _binding(case, LaunchNamespace.NCU),
        LaunchNamespace.NVBIT: _binding(case, LaunchNamespace.NVBIT),
        LaunchNamespace.SIMULATOR: _binding(case, LaunchNamespace.SIMULATOR),
    }
    selected_bindings.update(bindings or {})
    return evaluate_launch_crosswalk_fixture(
        canonical_inventory=installed[LaunchNamespace.CANONICAL],
        ncu_inventory=installed[LaunchNamespace.NCU],
        nvbit_inventory=installed[LaunchNamespace.NVBIT],
        simulator_inventory=installed[LaunchNamespace.SIMULATOR],
        canonical_candidates=supplied[LaunchNamespace.CANONICAL],
        ncu_candidates=supplied[LaunchNamespace.NCU],
        nvbit_candidates=supplied[LaunchNamespace.NVBIT],
        simulator_candidates=supplied[LaunchNamespace.SIMULATOR],
        ncu_binding=selected_bindings[LaunchNamespace.NCU],
        nvbit_binding=selected_bindings[LaunchNamespace.NVBIT],
        simulator_binding=selected_bindings[LaunchNamespace.SIMULATOR],
        **_schema_arguments(),
    )


def _verify(model: dict[str, Any]) -> None:
    verify_launch_crosswalk_fixture_model(model, **_schema_arguments())


def _reseal(model: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(model)
    changed.pop("model_seal_sha256", None)
    return seal_model(changed)


def _mutate_binding(
    binding: IndependentLaunchBinding, field: str, value: object
) -> IndependentLaunchBinding:
    model = binding.to_model()
    model[field] = value
    model.pop("model_seal_sha256")
    return IndependentLaunchBinding.from_model(seal_model(model))


def test_schema_bytes_are_runtime_pinned() -> None:
    assert sha256_bytes(SCHEMA_BYTES) == SCHEMA_SHA256
    assert json.loads(SCHEMA_BYTES)["$id"] == SCHEMA_ID


def test_production_authority_registries_are_empty_and_paths_block(
    tmp_path: Path,
) -> None:
    assert dict(PRODUCTION_LAUNCH_PARSER_AUTHORITIES) == {}
    assert dict(PRODUCTION_CROSS_NAMESPACE_PROVENANCE_AUTHORITIES) == {}
    assert dict(PRODUCTION_BINDING_AUTHORITIES) == {}

    production_path = tmp_path / "production-source.bin"
    production_path.write_bytes(b"unknown-production-source-format")
    payload, snapshot = RetainedPayloadSnapshot.read_path(production_path)
    source = SourceArtifactDescriptor(
        namespace=LaunchNamespace.NCU,
        artifact_kind="NCU_LAUNCH_SOURCE",
        artifact_sha256=sha256_bytes(payload),
        snapshot=snapshot,
    )
    with pytest.raises(EvaluatorAuthorityUnavailable) as parser_blocker:
        create_launch_inventory(
            source=source,
            source_payload=payload,
            candidates=(),
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        )
    assert (
        parser_blocker.value.code
        is AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE
    )

    case = _make_case(tmp_path, token="production-blockers")
    binding = _binding(case, LaunchNamespace.NCU)
    production_calls = (
        lambda: evaluate_launch_crosswalk(),
        lambda: verify_launch_inventory_model(
            case.inventories[LaunchNamespace.NCU],
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        ),
        lambda: verify_launch_crosswalk_model(
            _evaluate(case), **_schema_arguments()
        ),
        lambda: load_launch_crosswalk(
            canonical_model_bytes(_evaluate(case)), **_schema_arguments()
        ),
    )
    for call in production_calls:
        with pytest.raises(EvaluatorAuthorityUnavailable) as captured:
            call()
        assert (
            captured.value.code
            is AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE
        )

    with pytest.raises(EvaluatorAuthorityUnavailable) as binding_blocker:
        IndependentLaunchBinding.create(
            inventory=case.inventories[LaunchNamespace.NCU],
            canonical_launch=case.canonical_launch,
            selected_launch=case.records[LaunchNamespace.NCU][0],
            binding_authority_sha256=binding.binding_authority_sha256,
            inventory_schema_bytes=SCHEMA_BYTES,
            inventory_schema_snapshot=SCHEMA_SNAPSHOT,
        )
    assert (
        binding_blocker.value.code
        is AuthorityBlockerCode.BINDING_AUTHORITY_UNAVAILABLE
    )

    with pytest.raises(EvaluatorAuthorityUnavailable) as descriptor_blocker:
        SourceArtifactDescriptor.from_payload(
            namespace=LaunchNamespace.NCU,
            artifact_kind="NCU_LAUNCH_SOURCE",
            payload=payload,
            snapshot=snapshot,
        )
    assert (
        descriptor_blocker.value.code
        is AuthorityBlockerCode.LAUNCH_PARSER_AUTHORITY_UNAVAILABLE
    )


def test_unique_crosswalk_is_structurally_complete_but_production_unavailable(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    model = _evaluate(case)

    assert model["contract_mode"] == "FIXTURE_ONLY"
    assert model["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": [
            "FIXTURE_ONLY_CANONICAL_SELECTION",
            "FIXTURE_ONLY_NCU_BINDING",
            "FIXTURE_ONLY_NVBIT_BINDING",
            "FIXTURE_ONLY_SIMULATOR_BINDING",
        ],
    }
    assert set(model["namespaces"]) == {"canonical", "ncu", "nvbit", "simulator"}
    assert {
        name: selection["namespace"]
        for name, selection in model["namespaces"].items()
    } == {
        "canonical": "CANONICAL",
        "ncu": "NCU",
        "nvbit": "NVBIT",
        "simulator": "SIMULATOR",
    }
    _verify(model)
    _validator().validate(model)


def test_inventory_seals_complete_sorted_candidate_ids_and_count(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        source_specs={
            LaunchNamespace.NCU: (
                (SIGNATURE, 7, "COLD"),
                (OTHER_SIGNATURE, 2, "AUX"),
            )
        },
    )
    inventory = case.inventories[LaunchNamespace.NCU]
    records = inventory["candidate_records"]

    assert inventory["candidate_count"] == 2
    assert inventory["qualified_candidate_ids"] == sorted(
        record["identity"]["qualified_launch_sha256"] for record in records
    )
    assert [record["identity"]["qualified_launch_sha256"] for record in records] == (
        inventory["qualified_candidate_ids"]
    )
    assert inventory["source"]["artifact_sha256"] == case.sources[
        LaunchNamespace.NCU
    ].artifact_sha256
    verify_launch_inventory_fixture_model(
        inventory, schema_bytes=SCHEMA_BYTES, schema_snapshot=SCHEMA_SNAPSHOT
    )
    _validator().validate(inventory)


def test_inventory_and_crosswalk_seals_ignore_candidate_input_order(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        source_specs={
            LaunchNamespace.NCU: (
                (SIGNATURE, 7, "COLD"),
                (OTHER_SIGNATURE, 2, "AUX"),
            )
        },
    )
    namespace = LaunchNamespace.NCU
    reversed_inventory = create_launch_inventory_fixture(
        source=case.sources[namespace],
        source_payload=case.source_payloads[namespace],
        candidates=tuple(reversed(case.records[namespace])),
        schema_bytes=SCHEMA_BYTES,
        schema_snapshot=SCHEMA_SNAPSHOT,
    )
    assert reversed_inventory == case.inventories[namespace]

    forward = _evaluate(case)
    reverse = _evaluate(
        case, candidates={namespace: tuple(reversed(case.records[namespace]))}
    )
    assert forward == reverse
    assert forward["model_seal_sha256"] == reverse["model_seal_sha256"]


@pytest.mark.parametrize("mode", ["missing", "extra", "replacement"])
def test_supplied_candidate_set_must_equal_sealed_inventory(
    tmp_path: Path, mode: str
) -> None:
    case = _make_case(tmp_path)
    namespace = LaunchNamespace.NCU
    if mode == "missing":
        supplied: tuple[LaunchRecord, ...] = ()
    elif mode == "extra":
        extra = LaunchRecord.create(
            source=case.sources[namespace],
            signature=OTHER_SIGNATURE,
            launch_ordinal=1,
            launch_role="AUX",
        )
        supplied = case.records[namespace] + (extra,)
    else:
        replacement = _make_case(tmp_path, token="replacement")
        supplied = replacement.records[namespace]

    model = _evaluate(case, candidates={namespace: supplied})
    assert model["namespaces"]["ncu"]["outcome"] == {
        "status": "REJECTED",
        "reason_codes": ["NCU_INVENTORY_CANDIDATE_SET_MISMATCH"],
    }
    assert model["outcome"]["status"] == "REJECTED"
    _verify(model)


def test_cross_namespace_candidate_attempt_is_typed_and_schema_valid(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    model = _evaluate(
        case,
        candidates={LaunchNamespace.NCU: case.records[LaunchNamespace.NVBIT]},
    )
    assert model["namespaces"]["ncu"]["outcome"] == {
        "status": "REJECTED",
        "reason_codes": ["NCU_INVENTORY_CANDIDATE_SET_MISMATCH"],
    }
    _verify(model)
    _validator().validate(model)


def test_inventory_creation_rejects_caller_selected_parser_subset(
    tmp_path: Path,
) -> None:
    namespace = LaunchNamespace.NCU
    case = _make_case(
        tmp_path,
        source_specs={
            namespace: ((SIGNATURE, 0, "COLD"), (OTHER_SIGNATURE, 1, "AUX"))
        },
    )
    with pytest.raises(
        ImmutableArtifactError,
        match="differs from parser-derived complete inventory",
    ):
        create_launch_inventory_fixture(
            source=case.sources[namespace],
            source_payload=case.source_payloads[namespace],
            candidates=(case.records[namespace][0],),
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        )


@pytest.mark.parametrize(
    "target_namespace",
    [
        LaunchNamespace.CANONICAL,
        LaunchNamespace.NVBIT,
        LaunchNamespace.SIMULATOR,
    ],
)
def test_same_payload_cannot_be_relabeled_across_namespaces(
    tmp_path: Path, target_namespace: LaunchNamespace
) -> None:
    case = _make_case(tmp_path)
    original = case.sources[LaunchNamespace.NCU]
    payload = case.source_payloads[LaunchNamespace.NCU]
    relabeled = SourceArtifactDescriptor.from_fixture_payload(
        namespace=target_namespace,
        artifact_kind=_SOURCE_KINDS[target_namespace],
        payload=payload,
        snapshot=original.snapshot,
    )
    relabeled_record = LaunchRecord.create(
        source=relabeled,
        signature=SIGNATURE,
        launch_ordinal=0,
        launch_role="COLD",
    )
    with pytest.raises(ImmutableArtifactError, match="source namespace mismatch"):
        create_launch_inventory_fixture(
            source=relabeled,
            source_payload=payload,
            candidates=(relabeled_record,),
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        )


def test_inventory_creation_consumes_exact_source_bytes_and_source_kind(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    namespace = LaunchNamespace.NCU
    with pytest.raises(ImmutableArtifactError, match="does not match snapshot"):
        create_launch_inventory_fixture(
            source=case.sources[namespace],
            source_payload=case.source_payloads[namespace] + b"changed",
            candidates=case.records[namespace],
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        )
    with pytest.raises(ImmutableArtifactError, match="kind does not match namespace"):
        SourceArtifactDescriptor(
            namespace=namespace,
            artifact_kind="NVBIT_LAUNCH_SOURCE",
            artifact_sha256=case.sources[namespace].artifact_sha256,
            snapshot=case.sources[namespace].snapshot,
        )


def test_fixture_source_boolean_schema_version_is_rejected(tmp_path: Path) -> None:
    namespace = LaunchNamespace.NCU
    source_model = json.loads(
        fixture_launch_source_payload(
            namespace,
            (FixtureLaunchSpec(SIGNATURE, 0, "COLD"),),
        )
    )
    source_model["fixture_schema_version"] = True
    payload, snapshot = _retained_file(
        tmp_path / "boolean-source-version.json",
        canonical_model_bytes(source_model),
    )
    source = SourceArtifactDescriptor.from_fixture_payload(
        namespace=namespace,
        artifact_kind="NCU_LAUNCH_SOURCE",
        payload=payload,
        snapshot=snapshot,
    )
    record = LaunchRecord.create(
        source=source,
        signature=SIGNATURE,
        launch_ordinal=0,
        launch_role="COLD",
    )
    with pytest.raises(ImmutableArtifactError, match="schema version is invalid"):
        create_launch_inventory_fixture(
            source=source,
            source_payload=payload,
            candidates=(record,),
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        )


def test_wrong_namespace_inventory_is_rejected_before_selection(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    with pytest.raises(ImmutableArtifactError, match="wrong namespace"):
        _evaluate(
            case,
            inventories={
                LaunchNamespace.NCU: case.inventories[LaunchNamespace.NVBIT]
            },
        )


def test_missing_signature_is_typed_unavailable(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        source_specs={LaunchNamespace.NVBIT: ((OTHER_SIGNATURE, 0, "AUX"),)},
    )
    model = _evaluate(case)
    assert model["namespaces"]["nvbit"]["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": ["NVBIT_SIGNATURE_UNAVAILABLE"],
    }
    assert model["outcome"]["status"] == "UNAVAILABLE"
    _verify(model)


@pytest.mark.parametrize(
    "namespace",
    [LaunchNamespace.NCU, LaunchNamespace.NVBIT, LaunchNamespace.SIMULATOR],
)
@pytest.mark.parametrize("source_role", ["WARM", "EVICT"])
def test_unique_signature_never_bypasses_role_binding_authority(
    tmp_path: Path, namespace: LaunchNamespace, source_role: str
) -> None:
    case = _make_case(
        tmp_path,
        source_specs={namespace: ((SIGNATURE, 0, source_role),)},
    )
    model = _evaluate(case, bindings={namespace: None})
    key = namespace.value.lower()
    assert model["namespaces"][key]["name_dimension_match_count"] == 1
    assert model["namespaces"][key]["selected_launch"] is None
    assert model["namespaces"][key]["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": [f"{namespace.value}_BINDING_AUTHORITY_UNAVAILABLE"],
    }
    assert model["outcome"]["status"] == "UNAVAILABLE"
    _verify(model)
    _validator().validate(model)


@pytest.mark.parametrize(
    "namespace",
    [LaunchNamespace.NCU, LaunchNamespace.NVBIT, LaunchNamespace.SIMULATOR],
)
def test_duplicate_name_dimensions_require_inventory_bound_ordinal_role_selector(
    tmp_path: Path, namespace: LaunchNamespace
) -> None:
    case = _make_case(
        tmp_path,
        source_specs={
            namespace: ((SIGNATURE, 3, "COLD"), (SIGNATURE, 4, "WARM"))
        },
    )
    ambiguous = _evaluate(case, bindings={namespace: None})
    key = namespace.value.lower()
    assert ambiguous["namespaces"][key]["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": [f"{namespace.value}_BINDING_AUTHORITY_UNAVAILABLE"],
    }

    selected_launch = next(
        record
        for record in case.records[namespace]
        if record.identity.launch_ordinal == 3
        and record.identity.launch_role == "COLD"
    )
    binding = _binding(case, namespace, selected=selected_launch)
    selected = _evaluate(case, bindings={namespace: binding})
    assert selected["outcome"]["status"] == "UNAVAILABLE"
    assert selected["namespaces"][key]["outcome"] == {
        "status": "UNAVAILABLE",
        "reason_codes": [f"FIXTURE_ONLY_{namespace.value}_BINDING"],
    }
    assert selected["namespaces"][key]["name_dimension_match_count"] == 2
    assert selected["namespaces"][key]["selected_launch"]["identity"][
        "qualified_launch_sha256"
    ] == selected_launch.identity.qualified_launch_sha256
    _verify(selected)


def test_duplicate_qualified_candidate_identity_is_rejected_at_inventory_creation(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    namespace = LaunchNamespace.NCU
    launch = case.records[namespace][0]
    with pytest.raises(ImmutableArtifactError, match="qualified candidate identity collision"):
        create_launch_inventory_fixture(
            source=case.sources[namespace],
            source_payload=case.source_payloads[namespace],
            candidates=(launch, launch),
            schema_bytes=SCHEMA_BYTES,
            schema_snapshot=SCHEMA_SNAPSHOT,
        )


def test_unicode_must_be_nfc_before_identity_sealing(tmp_path: Path) -> None:
    with pytest.raises(ImmutableArtifactError, match="NFC-normalized"):
        LaunchSignature("é_kernel", (1, 1, 1), (1, 1, 1))

    case = _make_case(tmp_path)
    with pytest.raises(ImmutableArtifactError, match="NFC-normalized"):
        LaunchRecord.create(
            source=case.sources[LaunchNamespace.NCU],
            signature=SIGNATURE,
            launch_ordinal=0,
            launch_role="évict",
        )


def test_nfd_kernel_name_is_rejected_by_schema_and_runtime(tmp_path: Path) -> None:
    valid = _evaluate(_make_case(tmp_path))
    hostile = copy.deepcopy(valid)
    hostile["selection_signature"]["kernel_name"] = "é_kernel"
    hostile = _reseal(hostile)

    with pytest.raises(ValidationError):
        _validator().validate(hostile)
    with pytest.raises(ImmutableArtifactError, match="NFC-normalized"):
        _verify(hostile)


def test_binding_replay_against_replacement_inventory_is_rejected(tmp_path: Path) -> None:
    case = _make_case(
        tmp_path,
        source_specs={
            LaunchNamespace.NCU: ((SIGNATURE, 3, "COLD"), (SIGNATURE, 4, "WARM"))
        },
    )
    stale = _binding(case, LaunchNamespace.NCU, selected=case.records[LaunchNamespace.NCU][0])
    replacement = _make_case(
        tmp_path,
        token="replacement",
        source_specs={
            LaunchNamespace.NCU: ((SIGNATURE, 3, "COLD"), (SIGNATURE, 4, "WARM"))
        },
    )
    model = _evaluate(
        case,
        inventories={
            LaunchNamespace.NCU: replacement.inventories[LaunchNamespace.NCU]
        },
        candidates={LaunchNamespace.NCU: replacement.records[LaunchNamespace.NCU]},
        bindings={LaunchNamespace.NCU: stale},
    )
    assert model["namespaces"]["ncu"]["outcome"] == {
        "status": "REJECTED",
        "reason_codes": ["NCU_BINDING_CONTEXT_MISMATCH"],
    }


def test_binding_replay_after_role_sequence_change_is_rejected(tmp_path: Path) -> None:
    namespace = LaunchNamespace.NCU
    case = _make_case(
        tmp_path,
        source_specs={namespace: ((SIGNATURE, 3, "COLD"), (SIGNATURE, 4, "WARM"))},
    )
    stale = _binding(case, namespace, selected=case.records[namespace][0])
    changed_source, changed_payload = _source(
        tmp_path,
        namespace,
        "role-change",
        ((SIGNATURE, 3, "EVICT"), (SIGNATURE, 4, "WARM")),
    )
    changed_records = (
        LaunchRecord.create(
            source=changed_source,
            signature=SIGNATURE,
            launch_ordinal=3,
            launch_role="EVICT",
        ),
        LaunchRecord.create(
            source=changed_source,
            signature=SIGNATURE,
            launch_ordinal=4,
            launch_role="WARM",
        ),
    )
    changed_inventory = create_launch_inventory_fixture(
        source=changed_source,
        source_payload=changed_payload,
        candidates=changed_records,
        schema_bytes=SCHEMA_BYTES,
        schema_snapshot=SCHEMA_SNAPSHOT,
    )
    model = _evaluate(
        case,
        inventories={namespace: changed_inventory},
        candidates={namespace: changed_records},
        bindings={namespace: stale},
    )
    assert model["namespaces"]["ncu"]["outcome"]["status"] == "REJECTED"
    assert model["namespaces"]["ncu"]["outcome"]["reason_codes"] == [
        "NCU_BINDING_NO_MATCH"
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("inventory_model_seal_sha256", "f" * 64),
        ("source_artifact_sha256", "f" * 64),
        ("source_snapshot_identity_sha256", "f" * 64),
        ("launch_ordinal", 99),
        ("launch_role", "EVICT"),
        ("qualified_launch_sha256", "f" * 64),
        ("canonical_qualified_launch_sha256", "f" * 64),
        ("canonical_signature_sha256", "f" * 64),
    ],
)
def test_each_selector_context_field_is_anti_replay_bound(
    tmp_path: Path, field: str, replacement: object
) -> None:
    namespace = LaunchNamespace.NCU
    case = _make_case(
        tmp_path,
        source_specs={namespace: ((SIGNATURE, 3, "COLD"), (SIGNATURE, 4, "WARM"))},
    )
    binding = _binding(case, namespace, selected=case.records[namespace][0])
    changed = _mutate_binding(binding, field, replacement)
    model = _evaluate(case, bindings={namespace: changed})
    expected_reason = (
        "NCU_BINDING_NO_MATCH"
        if field in {"launch_ordinal", "launch_role"}
        else "NCU_BINDING_CONTEXT_MISMATCH"
    )
    assert model["namespaces"]["ncu"]["outcome"] == {
        "status": "REJECTED",
        "reason_codes": [expected_reason],
    }


def test_resealed_changed_binding_authority_is_rejected_by_schema_and_runtime(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    valid = _evaluate(case)
    hostile = copy.deepcopy(valid)
    binding = hostile["namespaces"]["ncu"]["independent_binding"]
    binding["binding_authority_sha256"] = "f" * 64
    hostile["namespaces"]["ncu"]["independent_binding"] = _reseal(binding)
    hostile = _reseal(hostile)

    with pytest.raises(ValidationError):
        _validator().validate(hostile)
    with pytest.raises(
        ImmutableArtifactError,
        match="binding authority SHA-256 is not pinned",
    ):
        _verify(hostile)


@pytest.mark.parametrize(
    ("role_field", "replacement"),
    [("canonical_role", "OTHER_TARGET"), ("source_role", "EVICT")],
)
def test_explicit_role_mapping_is_anti_replay_bound(
    tmp_path: Path, role_field: str, replacement: str
) -> None:
    namespace = LaunchNamespace.NCU
    case = _make_case(
        tmp_path,
        source_specs={namespace: ((SIGNATURE, 3, "COLD"), (SIGNATURE, 4, "WARM"))},
    )
    binding = _binding(case, namespace, selected=case.records[namespace][0])
    binding_model = binding.to_model()
    assert binding_model["role_mapping"] == {
        "mapping_kind": "EXACT_DECLARED_ROLE_MAPPING",
        "canonical_role": "TARGET",
        "source_role": "COLD",
    }
    binding_model["role_mapping"][role_field] = replacement
    binding_model.pop("model_seal_sha256")
    replayed = IndependentLaunchBinding.from_model(seal_model(binding_model))

    model = _evaluate(case, bindings={namespace: replayed})
    assert model["namespaces"]["ncu"]["outcome"] == {
        "status": "REJECTED",
        "reason_codes": ["NCU_BINDING_CONTEXT_MISMATCH"],
    }


def test_binding_cannot_be_replayed_under_another_namespace(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    ncu_binding = _binding(case, LaunchNamespace.NCU)
    model = _evaluate(case, bindings={LaunchNamespace.NVBIT: ncu_binding})
    assert model["namespaces"]["nvbit"]["outcome"] == {
        "status": "REJECTED",
        "reason_codes": ["NVBIT_BINDING_CONTEXT_MISMATCH"],
    }
    _verify(model)
    _validator().validate(model)


def test_crosswalk_contains_no_metric_values_and_preserves_disabled_boundary(
    tmp_path: Path,
) -> None:
    model = _evaluate(_make_case(tmp_path))
    forbidden = {
        "metric_value",
        "metric_values",
        "observed_value",
        "predicted_value",
        "target_value",
        "score",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(model)
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


def test_loader_rejects_duplicate_json_and_unknown_version(tmp_path: Path) -> None:
    model = _evaluate(_make_case(tmp_path))
    payload = canonical_model_bytes(model)
    assert load_launch_crosswalk_fixture(payload, **_schema_arguments()) == model

    duplicate = payload[:-1] + b',"schema_version":1}'
    with pytest.raises(ImmutableArtifactError, match="duplicate JSON key"):
        load_launch_crosswalk_fixture(duplicate, **_schema_arguments())

    changed = copy.deepcopy(model)
    changed["schema_version"] = 2
    changed = _reseal(changed)
    with pytest.raises(ImmutableArtifactError, match="schema_version mismatch"):
        load_launch_crosswalk_fixture(canonical_model_bytes(changed), **_schema_arguments())


def test_schema_and_runtime_reject_boolean_counts_and_terminal_newlines(
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path)
    valid = _evaluate(
        case,
        bindings={LaunchNamespace.NCU: _binding(case, LaunchNamespace.NCU)},
    )
    validator = _validator()
    validator.validate(valid)

    bad_version = copy.deepcopy(valid)
    bad_version["schema_version"] = True
    bad_version = _reseal(bad_version)
    with pytest.raises(ValidationError):
        validator.validate(bad_version)
    with pytest.raises(ImmutableArtifactError, match="schema_version"):
        _verify(bad_version)

    bad_binding_version = copy.deepcopy(valid)
    binding_model = bad_binding_version["namespaces"]["ncu"][
        "independent_binding"
    ]
    binding_model["binding_version"] = True
    bad_binding_version["namespaces"]["ncu"]["independent_binding"] = _reseal(
        binding_model
    )
    bad_binding_version = _reseal(bad_binding_version)
    with pytest.raises(ValidationError):
        validator.validate(bad_binding_version)
    with pytest.raises(ImmutableArtifactError, match="binding version"):
        _verify(bad_binding_version)

    bad_source_kind = copy.deepcopy(valid)
    binding_model = bad_source_kind["namespaces"]["ncu"]["independent_binding"]
    binding_model["source_artifact_kind"] = "NVBIT_LAUNCH_SOURCE"
    bad_source_kind["namespaces"]["ncu"]["independent_binding"] = _reseal(
        binding_model
    )
    bad_source_kind = _reseal(bad_source_kind)
    with pytest.raises(ValidationError):
        validator.validate(bad_source_kind)
    with pytest.raises(ImmutableArtifactError, match="source kind"):
        _verify(bad_source_kind)

    success_namespace_relabel = copy.deepcopy(valid)
    binding_model = success_namespace_relabel["namespaces"]["ncu"][
        "independent_binding"
    ]
    binding_model["namespace"] = "NVBIT"
    binding_model["source_artifact_kind"] = "NVBIT_LAUNCH_SOURCE"
    success_namespace_relabel["namespaces"]["ncu"][
        "independent_binding"
    ] = _reseal(binding_model)
    success_namespace_relabel = _reseal(success_namespace_relabel)
    with pytest.raises(ValidationError):
        validator.validate(success_namespace_relabel)
    with pytest.raises(ImmutableArtifactError, match="selection result mismatch"):
        _verify(success_namespace_relabel)

    bad_count = copy.deepcopy(valid)
    inventory = bad_count["namespaces"]["ncu"]["inventory"]
    inventory["candidate_count"] = True
    bad_count["namespaces"]["ncu"]["inventory"] = _reseal(inventory)
    bad_count = _reseal(bad_count)
    with pytest.raises(ValidationError):
        validator.validate(bad_count)
    with pytest.raises(ImmutableArtifactError, match="candidate_count"):
        _verify(bad_count)

    bad_hash = copy.deepcopy(valid)
    bad_hash["namespaces"]["ncu"]["selected_launch"]["identity"][
        "qualified_launch_sha256"
    ] += "\n"
    bad_hash = _reseal(bad_hash)
    with pytest.raises(ValidationError):
        validator.validate(bad_hash)
    with pytest.raises(ImmutableArtifactError, match="lowercase 64-hex"):
        _verify(bad_hash)

    bad_role = copy.deepcopy(valid)
    bad_role["namespaces"]["ncu"]["selected_launch"]["identity"][
        "launch_role"
    ] += "\n"
    bad_role = _reseal(bad_role)
    with pytest.raises(ValidationError):
        validator.validate(bad_role)
    with pytest.raises(ImmutableArtifactError, match="forbidden control"):
        _verify(bad_role)


def test_rejected_and_unavailable_reason_vocabularies_are_disjoint(
    tmp_path: Path,
) -> None:
    valid = _evaluate(_make_case(tmp_path))
    for status, reason in (
        ("UNAVAILABLE", "NCU_BINDING_CONTEXT_MISMATCH"),
        ("REJECTED", "NCU_SIGNATURE_UNAVAILABLE"),
        ("UNAVAILABLE", "NVBIT_SIGNATURE_UNAVAILABLE"),
        ("REJECTED", "NVBIT_BINDING_CONTEXT_MISMATCH"),
    ):
        invalid = copy.deepcopy(valid)
        invalid["namespaces"]["ncu"]["selected_launch"] = None
        invalid["namespaces"]["ncu"]["outcome"] = {
            "status": status,
            "reason_codes": [reason],
        }
        invalid["outcome"] = {"status": status, "reason_codes": [reason]}
        invalid = _reseal(invalid)

        with pytest.raises(ValidationError):
            _validator().validate(invalid)
        with pytest.raises(ImmutableArtifactError, match="reason vocabulary"):
            _verify(invalid)


def test_schema_is_closed_against_metrics_and_identity_shape_changes(
    tmp_path: Path,
) -> None:
    valid = _evaluate(_make_case(tmp_path))
    validator = _validator()
    invalid_models: list[dict[str, Any]] = []

    extra = copy.deepcopy(valid)
    extra["metric_values"] = {"gpc__cycles_elapsed.max": 42}
    invalid_models.append(extra)

    nested_extra = copy.deepcopy(valid)
    nested_extra["namespaces"]["ncu"]["selected_launch"]["metric_value"] = 42
    invalid_models.append(nested_extra)

    malformed_hash = copy.deepcopy(valid)
    malformed_hash["namespaces"]["ncu"]["selected_launch"]["identity"][
        "source_artifact_sha256"
    ] = "B" * 64
    invalid_models.append(malformed_hash)

    identity_swap = copy.deepcopy(valid)
    identity_swap["namespaces"]["ncu"]["selected_launch"]["identity"][
        "namespace"
    ] = "NVBIT"
    invalid_models.append(identity_swap)

    for invalid in invalid_models:
        with pytest.raises(ValidationError):
            validator.validate(invalid)
