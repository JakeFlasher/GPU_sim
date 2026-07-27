from __future__ import annotations

import copy
import json
import unicodedata
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from tools.local_labels import immutable_artifact as immutable
from tools.local_labels import m5_authority_ancestry_audit as audit
from tools.local_labels import m5_calibration_controller as m5


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_TOOL = Path(audit.__file__).resolve()
AUDIT_SCHEMA = (
    REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "m5_authority_ancestry_audit.v1.schema.json"
)
SOURCE_REGISTRY = Path(__file__).with_name("m5_source_surface_registry.json")
SOURCE_REGISTRY_SCHEMA = (
    REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "m5_source_surface_registry.v1.schema.json"
)
QUARANTINE_PINS = (
    (
        "S001",
        "plans/sm86_calibration_microbench_design_fable5_prompt.md",
        "8bc769b0b15f5a60fe8b12d52bf91c3aca8282999057131ff1324df257b13d93",
    ),
    (
        "S002",
        "plans/sm86_calibration_microbench_design_fable5_answer.md",
        "f2fe00cfe045c251f97d40c83dbb08ace731bfdec230a44907c0e5112620f20c",
    ),
    (
        "S003",
        "plans/sm86_cp_async_timing_probe_fable5_prompt.md",
        "1704f69bd85abb9c2042ff20b4f6173ceda8476045e6bca08cf52d75fe282402",
    ),
    (
        "S004",
        "plans/sm86_cp_async_timing_probe_fable5_answer.md",
        "a90eaec5b7c7c4d78c662c1273b68e7f0b6b0087cce81ae60f514095288d115f",
    ),
)


def _artifact(
    identity_id: str, payload: bytes, media_type: str = "text/markdown"
) -> dict[str, Any]:
    digest = immutable.sha256_bytes(payload)
    artifact_id = f"{identity_id}-input"
    return {
        "artifact_id": artifact_id,
        "relative_path": f"artifacts/{digest}/{artifact_id}",
        "sha256": digest,
        "size_bytes": len(payload),
        "media_type": media_type,
    }


def _fixture() -> tuple[dict[str, Any], bytes, dict[str, tuple[dict[str, Any], bytes]]]:
    design_input_id = "synthetic-design-input"
    payload = b"Synthetic offline design used only to test mechanical audit closure.\n"
    artifact = _artifact(design_input_id, payload)
    quantities = copy.deepcopy(m5._quantity_authority_model()["quantities"])
    for quantity in quantities:
        quantity["meaning"] = f"Synthetic meaning for {quantity['quantity_id']}"
        quantity["conditioning"]["variant"] = f"synthetic.{quantity['quantity_id']}"
    execution_order = ["a", "c", "e", "b", "d", "f"]
    scientific_rank = ["e", "c", "a", "f", "d", "b"]
    synthetic_dependencies = {
        "a": [],
        "c": ["a"],
        "e": [],
        "b": ["c"],
        "d": ["a", "e"],
        "f": ["c", "e"],
    }
    for quantity in quantities:
        quantity["upstream_family_ids"] = (
            synthetic_dependencies[quantity["producer"]["family_id"]]
            if quantity["producer"]["kind"] == "PROBE_FAMILY"
            else []
        )
    order = {family: index for index, family in enumerate(execution_order)}
    quantities.sort(
        key=lambda quantity: (
            order.get(quantity["producer"].get("family_id"), 6),
            quantity["quantity_id"],
        )
    )
    provenance = {
        "design_input_bindings": [
            {
                "identity_id": design_input_id,
                "sha256": artifact["sha256"],
                "size_bytes": artifact["size_bytes"],
                "media_type": artifact["media_type"],
                "role": "SCIENTIFIC_DESIGN_INPUT",
            }
        ],
        "scientific_field_sources": [
            {
                "field_pointer": pointer,
                "design_input_identity_ids": [design_input_id],
            }
            for pointer in (
                "/execution_order",
                "/scientific_rank",
                *(f"/quantities/{index}" for index in range(len(quantities))),
            )
        ],
    }
    authority = {
        "execution_order": execution_order,
        "scientific_rank": scientific_rank,
        "quantities": quantities,
        "consumer_binding_registry": [
            {
                "quantity_id": quantity["quantity_id"],
                "consumer_binding": quantity["consumer_binding"],
                "authorized_gap_kinds": quantity["authorized_gap_kinds"],
                "research_gap_only": True,
            }
            for quantity in quantities
        ],
        "gap_mapping": {
            "mapping_version": 1,
            "typed_gap_kinds": sorted(m5.TYPED_GAPS),
            "consumer_rule": "RETAIN_AS_TYPED_RESEARCH_GAP",
            "ncu_selection_forbidden": True,
        },
        "provenance": provenance,
    }
    raw = immutable.pretty_json_bytes(authority)
    return authority, raw, {design_input_id: (artifact, payload)}


def _build(
    authority: dict[str, Any],
    raw: bytes,
    design_inputs: dict[str, tuple[dict[str, Any], bytes]],
    *,
    registry_payload: bytes | None = None,
    registry_schema_payload: bytes | None = None,
    audit_tool_payload: bytes | None = None,
    audit_schema_payload: bytes | None = None,
    legacy_authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_payload = (
        AUDIT_TOOL.read_bytes()
        if audit_tool_payload is None
        else audit_tool_payload
    )
    implementation = m5._retained_ancestry_audit_module(tool_payload)
    try:
        return implementation.build_audit(
            authority=authority,
            authority_raw=raw,
            design_inputs=design_inputs,
            source_registry_payload=(
                SOURCE_REGISTRY.read_bytes()
                if registry_payload is None
                else registry_payload
            ),
            source_registry_schema_payload=(
                SOURCE_REGISTRY_SCHEMA.read_bytes()
                if registry_schema_payload is None
                else registry_schema_payload
            ),
            audit_schema_payload=(
                AUDIT_SCHEMA.read_bytes()
                if audit_schema_payload is None
                else audit_schema_payload
            ),
            legacy_authority=(
                m5._quantity_authority_model()
                if legacy_authority is None
                else legacy_authority
            ),
        )
    except implementation.AncestryAuditError as exc:
        raise audit.AncestryAuditError(str(exc)) from exc


def _replace_design_input(
    authority: dict[str, Any],
    design_inputs: dict[str, tuple[dict[str, Any], bytes]],
    payload: bytes,
    media_type: str = "text/markdown",
) -> None:
    identity_id = next(iter(design_inputs))
    artifact = _artifact(identity_id, payload, media_type)
    design_inputs[identity_id] = (artifact, payload)
    authority["provenance"]["design_input_bindings"][0].update(
        sha256=artifact["sha256"],
        size_bytes=artifact["size_bytes"],
        media_type=artifact["media_type"],
    )


def test_clean_fixture_audit_is_deterministic_closed_and_noncredit() -> None:
    authority, raw, design_inputs = _fixture()

    first = _build(authority, raw, design_inputs)
    second = _build(authority, raw, design_inputs)

    assert first == second
    assert first["verdict"] == "NO_LISTED_REVIVAL_DETECTED"
    assert first["mechanical_non_revival_only"] is True
    assert first["semantic_review_required"] is True
    assert first["external_credit"] == "NONE"
    assert first["checks"]["retained_tool_execution_bound"] is True
    assert first["checks"]["variable_scientific_fields_scanned"] is True
    assert first["checks"]["legacy_dependency_edges_absent"] is True
    assert audit._NORMALIZATION_POLICY["unicode_database_version"] == (
        unicodedata.unidata_version
    )
    assert first["audit_tool_sha256"] == immutable.sha256_file(AUDIT_TOOL)[0]
    assert first["audit_schema_sha256"] == immutable.sha256_file(AUDIT_SCHEMA)[0]
    assert first["source_registry_schema_sha256"] == immutable.sha256_file(
        SOURCE_REGISTRY_SCHEMA
    )[0]
    schema = json.loads(AUDIT_SCHEMA.read_bytes())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(first)


def test_legacy_execution_order_is_rejected_independently() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["execution_order"] = copy.deepcopy(
        m5._quantity_authority_model()["execution_order"]
    )

    with pytest.raises(audit.AncestryAuditError, match="execution order"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_legacy_scientific_rank_is_rejected_independently() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["scientific_rank"] = copy.deepcopy(
        m5._quantity_authority_model()["scientific_rank"]
    )

    with pytest.raises(audit.AncestryAuditError, match="scientific rank"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_one_prose_field_change_does_not_cleanse_quantity_contract() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["quantities"][0] = copy.deepcopy(
        m5._quantity_authority_model()["quantities"][0]
    )
    authority["quantities"][0]["meaning"] = "Cosmetically rewritten prose"
    authority["quantities"][0]["upstream_family_ids"] = []

    with pytest.raises(audit.AncestryAuditError, match="quantity contract"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


@pytest.mark.parametrize(("quarantine_id", "relative_path", "expected_sha256"), QUARANTINE_PINS)
def test_each_renamed_exact_quarantined_input_is_rejected(
    quarantine_id: str, relative_path: str, expected_sha256: str
) -> None:
    authority, _raw, design_inputs = _fixture()
    payload = (REPO_ROOT / relative_path).read_bytes()
    assert immutable.sha256_bytes(payload) == expected_sha256, quarantine_id
    _replace_design_input(authority, design_inputs, payload)

    with pytest.raises(audit.AncestryAuditError, match="exactly revives"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


@pytest.mark.parametrize(
    ("payload", "media_type"),
    [
        ("FaBlE-5".encode(), "text/markdown"),
        ("S–001".encode(), "text/markdown"),
        (b"S 001", "text/markdown"),
        (b"N.C.U", "text/markdown"),
        (b"gpc __cycles_elapsed.max", "text/markdown"),
        ("2.8×".encode(), "text/markdown"),
        (b"accuracy\r\n  wall", "text/markdown"),
        (b"GPC__CYCLES_ELAPSED.MAX", "text/markdown"),
        ("read-only attribution pass".encode(), "text/markdown"),
        (b"sm86_calibration_microbench_design_fable5_prompt.md", "text/markdown"),
        ("n​cu".encode(), "text/markdown"),
        ("n̸cu".encode(), "text/markdown"),
        ("Nsight—Compute".encode(), "text/markdown"),
        ("Nsight​Compute".encode(), "text/markdown"),
        ("accuracy​wall".encode(), "text/markdown"),
        (b'{"source":"\\u006e\\u0063\\u0075"}', "application/json"),
        (
            b'{"metric":"gpc\\u005f\\u005fcycles_elapsed.max"}',
            "application/json",
        ),
    ],
)
def test_normalized_json_or_metric_contamination_is_rejected(
    payload: bytes, media_type: str
) -> None:
    authority, _raw, design_inputs = _fixture()
    _replace_design_input(authority, design_inputs, payload, media_type)

    with pytest.raises(
        audit.AncestryAuditError, match="fragment|identifier|target-metric"
    ):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_variable_authority_target_token_is_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["quantities"][0]["meaning"] = (
        "Derived from N.C.U gpc __cycles_elapsed.max target labels"
    )

    with pytest.raises(
        audit.AncestryAuditError, match="variable scientific.*target-metric"
    ):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_variable_authority_historical_numeric_fragment_is_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["quantities"][0]["agreement_limit_fraction"] = 0.358

    with pytest.raises(
        audit.AncestryAuditError, match="variable scientific.*fragment"
    ):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_fixed_json_policy_key_is_not_treated_as_scientific_content() -> None:
    authority, _raw, design_inputs = _fixture()
    _replace_design_input(
        authority,
        design_inputs,
        b'{"ncu_selection_forbidden":true,"description":"synthetic input"}',
        "application/json",
    )

    model = _build(
        authority, immutable.pretty_json_bytes(authority), design_inputs
    )

    assert model["checks"]["variable_scientific_fields_scanned"] is True
    assert model["checks"]["normalized_forbidden_tokens_absent"] is True


def test_json_target_token_key_is_still_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    _replace_design_input(
        authority,
        design_inputs,
        b'{"gpc__cycles_elapsed.max":"synthetic"}',
        "application/json",
    )

    with pytest.raises(audit.AncestryAuditError, match="target-metric"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_legacy_dependency_edges_are_rejected_independently() -> None:
    authority, _raw, design_inputs = _fixture()
    legacy_dependencies = {
        "a": [],
        "b": [],
        "c": [],
        "d": ["b"],
        "e": [],
        "f": ["b", "d"],
    }
    for quantity in authority["quantities"]:
        if quantity["producer"]["kind"] == "PROBE_FAMILY":
            quantity["upstream_family_ids"] = legacy_dependencies[
                quantity["producer"]["family_id"]
            ]

    with pytest.raises(audit.AncestryAuditError, match="dependency edges"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


@pytest.mark.parametrize(
    ("upstream_family_id", "producer_family_id"),
    audit.LEGACY_DEPENDENCY_EDGES,
)
def test_each_legacy_dependency_edge_is_rejected_independently(
    upstream_family_id: str, producer_family_id: str
) -> None:
    authority, _raw, design_inputs = _fixture()
    for quantity in authority["quantities"]:
        if quantity["producer"]["kind"] == "PROBE_FAMILY":
            quantity["upstream_family_ids"] = (
                [upstream_family_id]
                if quantity["producer"]["family_id"] == producer_family_id
                else []
            )

    with pytest.raises(audit.AncestryAuditError, match="dependency edges"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_legacy_dependency_edges_cannot_be_cleansed_by_an_extra_edge() -> None:
    authority, _raw, design_inputs = _fixture()
    contaminated_dependencies = {
        "a": [],
        "c": ["a"],
        "e": [],
        "b": [],
        "d": ["b"],
        "f": ["b", "d"],
    }
    for quantity in authority["quantities"]:
        if quantity["producer"]["kind"] == "PROBE_FAMILY":
            quantity["upstream_family_ids"] = contaminated_dependencies[
                quantity["producer"]["family_id"]
            ]

    with pytest.raises(audit.AncestryAuditError, match="dependency edges"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_missing_scientific_field_coverage_is_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["provenance"]["scientific_field_sources"].pop()

    with pytest.raises(audit.AncestryAuditError, match="coverage is incomplete"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_unbound_retained_design_input_is_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    payload = b"Additional undeclared synthetic input.\n"
    design_inputs["undeclared-input"] = (_artifact("undeclared-input", payload), payload)

    with pytest.raises(audit.AncestryAuditError, match="identity sets differ"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_distinct_design_input_identities_cannot_alias_bytes() -> None:
    authority, _raw, design_inputs = _fixture()
    first_id = next(iter(design_inputs))
    first_artifact, payload = design_inputs[first_id]
    alias_id = "synthetic-design-alias"
    alias_artifact = copy.deepcopy(first_artifact)
    alias_artifact["artifact_id"] = f"{alias_id}-input"
    alias_artifact["relative_path"] = (
        f"artifacts/{alias_artifact['sha256']}/{alias_artifact['artifact_id']}"
    )
    design_inputs[alias_id] = (alias_artifact, payload)
    authority["provenance"]["design_input_bindings"].append(
        {
            "identity_id": alias_id,
            "sha256": alias_artifact["sha256"],
            "size_bytes": alias_artifact["size_bytes"],
            "media_type": alias_artifact["media_type"],
            "role": "SCIENTIFIC_DESIGN_INPUT",
        }
    )
    authority["provenance"]["scientific_field_sources"][0][
        "design_input_identity_ids"
    ].append(alias_id)

    with pytest.raises(audit.AncestryAuditError, match="alias identical artifact bytes"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


@pytest.mark.parametrize(("quarantine_id", "relative_path", "expected_sha256"), QUARANTINE_PINS)
def test_each_source_registry_quarantine_pin_is_independent(
    quarantine_id: str, relative_path: str, expected_sha256: str
) -> None:
    authority, raw, design_inputs = _fixture()
    registry = json.loads(SOURCE_REGISTRY.read_bytes())
    entry = next(entry for entry in registry["entries"] if entry["path"] == relative_path)
    assert entry["quarantine_id"] == quarantine_id
    assert entry["sha256"] == expected_sha256
    entry["sha256"] = "0" * 64

    with pytest.raises(audit.AncestryAuditError, match="schema violation|pin changed"):
        _build(
            authority,
            raw,
            design_inputs,
            registry_payload=immutable.pretty_json_bytes(registry),
        )


def test_source_registry_schema_and_duplicate_paths_are_enforced() -> None:
    authority, raw, design_inputs = _fixture()
    registry = json.loads(SOURCE_REGISTRY.read_bytes())
    registry["entries"].append(copy.deepcopy(registry["entries"][0]))

    with pytest.raises(audit.AncestryAuditError, match="duplicates path"):
        _build(
            authority,
            raw,
            design_inputs,
            registry_payload=immutable.pretty_json_bytes(registry),
        )

    registry = json.loads(SOURCE_REGISTRY.read_bytes())
    registry["collection_gate"] = "OPEN"
    with pytest.raises(audit.AncestryAuditError, match="root contract changed"):
        _build(
            authority,
            raw,
            design_inputs,
            registry_payload=immutable.pretty_json_bytes(registry),
        )


def test_direct_unbound_audit_execution_fails_closed() -> None:
    authority, raw, design_inputs = _fixture()

    with pytest.raises(audit.AncestryAuditError, match="execution binding is absent"):
        audit.build_audit(
            authority=authority,
            authority_raw=raw,
            design_inputs=design_inputs,
            source_registry_payload=SOURCE_REGISTRY.read_bytes(),
            source_registry_schema_payload=SOURCE_REGISTRY_SCHEMA.read_bytes(),
            audit_schema_payload=AUDIT_SCHEMA.read_bytes(),
            legacy_authority=m5._quantity_authority_model(),
        )


def test_retained_audit_has_no_live_nonstdlib_execution_dependency() -> None:
    source = AUDIT_TOOL.read_text(encoding="utf-8")

    assert "import jsonschema" not in source
    assert "import immutable_artifact" not in source
    assert "from tools.local_labels" not in source


@pytest.mark.parametrize(
    ("path", "field", "value", "message"),
    [
        (
            "plans/fable5_consultation_and_response/01-m5-rule1-provenance.fable5.response.txt",
            "lifecycle",
            "active",
            "quarantined ruling",
        ),
        (
            "workloads/simbench_bw/schemas/m5_calibration_run.v1.schema.json",
            "lifecycle",
            "active",
            "quarantined v1 schema",
        ),
        (
            "tools/local_labels/fit_calibration_head.py",
            "category",
            "producer",
            "forbidden substitute",
        ),
        (
            "tools/local_labels/m5_calibration_collector.py",
            "lifecycle",
            "active",
            "disabled M5 collector",
        ),
    ],
)
def test_source_registry_conditional_classifications_are_enforced(
    path: str, field: str, value: str, message: str
) -> None:
    authority, raw, design_inputs = _fixture()
    registry = json.loads(SOURCE_REGISTRY.read_bytes())
    entry = next(row for row in registry["entries"] if row["path"] == path)
    entry[field] = value

    with pytest.raises(audit.AncestryAuditError, match=message):
        _build(
            authority,
            raw,
            design_inputs,
            registry_payload=immutable.pretty_json_bytes(registry),
        )


def test_audit_policy_digest_binds_tool_schema_tokens_and_legacy_fingerprints() -> None:
    authority, raw, design_inputs = _fixture()
    baseline = _build(authority, raw, design_inputs)
    changed_tool = _build(
        authority,
        raw,
        design_inputs,
        audit_tool_payload=AUDIT_TOOL.read_bytes() + b"\n",
    )
    changed_schema = _build(
        authority,
        raw,
        design_inputs,
        audit_schema_payload=AUDIT_SCHEMA.read_bytes() + b"\n",
    )
    legacy = copy.deepcopy(m5._quantity_authority_model())
    legacy["quantities"][0]["conditioning"]["variant"] = "changed-policy-fixture"
    changed_legacy = _build(
        authority,
        raw,
        design_inputs,
        legacy_authority=legacy,
    )

    assert len(
        {
            baseline["audit_policy_sha256"],
            changed_tool["audit_policy_sha256"],
            changed_schema["audit_policy_sha256"],
            changed_legacy["audit_policy_sha256"],
        }
    ) == 4


def test_audit_schema_rejects_extra_properties_at_new_nested_boundaries() -> None:
    authority, raw, design_inputs = _fixture()
    model = _build(authority, raw, design_inputs)
    schema = json.loads(AUDIT_SCHEMA.read_bytes())
    validator = jsonschema.Draft202012Validator(schema)

    for path in (("root",), ("checks",), ("design_input",), ("artifact",)):
        candidate = copy.deepcopy(model)
        if path == ("root",):
            candidate["unexpected"] = True
        elif path == ("checks",):
            candidate["checks"]["unexpected"] = True
        elif path == ("design_input",):
            candidate["design_inputs"][0]["unexpected"] = True
        else:
            candidate["design_inputs"][0]["artifact"]["unexpected"] = True
        assert list(validator.iter_errors(candidate)), path


@pytest.mark.parametrize(
    ("payload", "media_type", "message"),
    [
        (b"", "text/markdown", "empty"),
        (b"x" * (audit.MAX_DESIGN_INPUT_BYTES + 1), "text/markdown", "oversized"),
        (b"synthetic\0binary", "text/markdown", "contains NUL"),
        (b"\xff", "text/markdown", "canonical UTF-8"),
        (b"synthetic", "application/octet-stream", "unsupported media type"),
    ],
)
def test_invalid_design_input_text_fails_closed(
    payload: bytes, media_type: str, message: str
) -> None:
    authority, _raw, design_inputs = _fixture()
    _replace_design_input(authority, design_inputs, payload, media_type)

    with pytest.raises(audit.AncestryAuditError, match=message):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_duplicate_signed_design_input_binding_is_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["provenance"]["design_input_bindings"].append(
        copy.deepcopy(authority["provenance"]["design_input_bindings"][0])
    )

    with pytest.raises(audit.AncestryAuditError, match="invalid or duplicated"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_duplicate_scientific_field_pointer_is_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["provenance"]["scientific_field_sources"].append(
        copy.deepcopy(authority["provenance"]["scientific_field_sources"][0])
    )

    with pytest.raises(audit.AncestryAuditError, match="invalid or duplicated"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)


def test_extra_scientific_field_pointer_is_rejected() -> None:
    authority, _raw, design_inputs = _fixture()
    authority["provenance"]["scientific_field_sources"].append(
        {
            "field_pointer": "/consumer_binding_registry",
            "design_input_identity_ids": [next(iter(design_inputs))],
        }
    )

    with pytest.raises(audit.AncestryAuditError, match="extra fields"):
        _build(authority, immutable.pretty_json_bytes(authority), design_inputs)
