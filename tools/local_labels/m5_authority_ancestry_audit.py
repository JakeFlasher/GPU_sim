"""Deterministic mechanical non-revival audit for M5 quantity authorities."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping


AUDIT_ID = "m5-quantity-authority-ancestry-audit"
AUDIT_MEDIA_TYPE = "application/vnd.m5.authority-ancestry-audit+json"
AUDIT_POLICY_VERSION = 1
SOURCE_REGISTRY_SCHEMA_ID = (
    "https://gpu-simulator.invalid/schemas/m5_source_surface_registry.v1.schema.json"
)
MAX_DESIGN_INPUT_BYTES = 1 << 20
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1
_EXECUTING_TOOL_SHA256: str | None = None
LEGACY_DEPENDENCY_SOURCE_COMMIT = (
    "97ad4e8c07b3b490f388e6590ad8d1fcc0e3ef78"
)
LEGACY_DEPENDENCY_EDGES = (
    ("b", "d"),
    ("b", "f"),
    ("d", "f"),
)
QUARANTINED_ANCESTRY = {
    "S001": (
        "plans/sm86_calibration_microbench_design_fable5_prompt.md",
        "8bc769b0b15f5a60fe8b12d52bf91c3aca8282999057131ff1324df257b13d93",
    ),
    "S002": (
        "plans/sm86_calibration_microbench_design_fable5_answer.md",
        "f2fe00cfe045c251f97d40c83dbb08ace731bfdec230a44907c0e5112620f20c",
    ),
    "S003": (
        "plans/sm86_cp_async_timing_probe_fable5_prompt.md",
        "1704f69bd85abb9c2042ff20b4f6173ceda8476045e6bca08cf52d75fe282402",
    ),
    "S004": (
        "plans/sm86_cp_async_timing_probe_fable5_answer.md",
        "a90eaec5b7c7c4d78c662c1273b68e7f0b6b0087cce81ae60f514095288d115f",
    ),
}
_TEXT_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "text/markdown",
        "text/plain",
        "text/x-cuda",
        "text/x-python",
    }
)
_JSON_FIXED_POLICY_KEYS = frozenset({"ncu_selection_forbidden"})
_SOURCE_REGISTRY_CLOSURE_ROOTS = (
    "CLAUDE.md",
    "arch_profiles",
    "data/local_labels",
    "data/pchase",
    "docs",
    "eval/frozen_labels",
    "eval/guards",
    "plans",
    "sim",
    "tools/local_labels",
    "tools/nvbit_tracer",
    "tools/pchase",
    "tools/producer_retirement.py",
    "tools/sass_control",
    "workloads/simbench_bw",
)
_SOURCE_CATEGORIES = frozenset(
    {
        "collector",
        "documentation",
        "downstream_consumer",
        "forbidden_substitute",
        "guard",
        "legacy_retired",
        "pinned_input",
        "producer",
        "runtime_dependency",
        "schema",
        "source_inventory",
        "test",
        "trust_registry",
    }
)
_SOURCE_LIFECYCLES = frozenset(
    {"active", "disabled", "historical", "quarantined"}
)
_SOURCE_PATH = re.compile(
    r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9_.+/-]{1,255}$"
)
_M5_RULING_PATH = re.compile(
    r"^plans/fable5_consultation_and_response/01-m5-rule1-provenance"
    r"(?:--final\.merged\.prompt|\.fable5\.response)\.txt$"
)
_V1_CALIBRATION_SCHEMA_PATH = re.compile(
    r"^workloads/simbench_bw/schemas/m5_calibration_"
    r"(?:run|result|done)\.v1\.schema\.json$"
)
_FORBIDDEN_SUBSTITUTE_PATH = re.compile(
    r"^(?:data/local_labels/nvml_(?:reference_operating_point|short_burst_anchor)_sm_86\.json"
    r"|eval/frozen_labels/sm86_m1_bundle_v21\.json"
    r"|tools/local_labels/(?:calibrate_nvml_.*|fit_.*calibration_head.*"
    r"|sm120_calibration_head_historical_only_scanner\.py"
    r"|test_calibration_head_v5_schema\.py)"
    r"|workloads/simbench_bw/(?:nvml_covariate_acceptance\.toml"
    r"|nvml_covariate_join\.py|nvml_metadata_verifier\.py"
    r"|schemas/(?:calibration_evaluation_.*|calibration_head_.*"
    r"|nvml_covariate\.schema\.json"
    r"|nvml_reference_operating_point\.schema\.json)))$"
)
_FORBIDDEN_RAW_FRAGMENTS = (
    "fable 5",
    "fable5",
    "nsight compute",
    "2.8x",
    "0.358",
    "13.96 pp",
    "37.5%",
    "62.5%",
    "55%",
    "668 b/cy",
    "accuracy wall",
    "simulator error attribution",
    "predicted/measured",
    "read only attribution pass",
    "load-bearing rank",
)
_METRIC_NAMESPACE = re.compile(
    r"(?<![a-z0-9])(?:dram|fbpa|gpc|gpu|l1tex|launch|lts|sass|sm|smsp)(?:[^a-z0-9]*?_){2}[^a-z0-9]*[a-z0-9][a-z0-9_.]*"
)
_NCU_TOKEN = re.compile(
    r"(?<![a-z0-9])n[^a-z0-9]*c[^a-z0-9]*u(?![a-z0-9])"
)
_QUARANTINE_ID_TOKEN = re.compile(
    r"(?<![a-z0-9])s[^a-z0-9]*0[^a-z0-9]*0[^a-z0-9]*[1-4](?![a-z0-9])"
)
_NORMALIZATION_POLICY = {
    "unicode_form": "NFKC",
    "unicode_database_version": unicodedata.unidata_version,
    "casefold": True,
    "identifier_drop_unicode_categories": ["Cf", "Me", "Mn"],
    "token_separate_unicode_categories": ["Cf", "Me", "Mn"],
    "unicode_dash_category": "Pd",
    "token_separators": (
        "whitespace, underscore, slash, backslash, and punctuation except dot/percent"
    ),
    "multiplication_sign": "x",
    "json_scan": (
        "recursively decoded keys except fixed policy keys, plus scalar values; "
        "raw serialization excluded"
    ),
    "fixed_json_policy_keys": sorted(_JSON_FIXED_POLICY_KEYS),
    "authority_scan": "variable scientific scalar values only",
}


class AncestryAuditError(RuntimeError):
    """A mechanical M5 ancestry audit could not produce a non-revival verdict."""


def _fail(message: str) -> None:
    raise AncestryAuditError(f"M5_RULE1_PREREGISTRATION_ANCESTRY: {message}")


def bind_executing_tool_sha256(digest: str) -> None:
    """Bind this module to the exact source bytes compiled by its retained loader."""

    global _EXECUTING_TOOL_SHA256
    if _EXECUTING_TOOL_SHA256 is not None:
        _fail("retained audit-tool execution binding was already set")
    if not _is_sha256(digest):
        _fail("retained audit-tool execution binding is not a SHA-256")
    _EXECUTING_TOOL_SHA256 = digest


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_nonfinite(value: Any) -> None:
    if type(value) is float:
        if not math.isfinite(value):
            _fail("non-finite JSON number")
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            _fail("JSON float exceeds the exact safe range")
    if type(value) is int and abs(value) > _MAX_SAFE_JSON_INTEGER:
        _fail("JSON integer exceeds the exact safe range")
    if type(value) is str and any(
        0xD800 <= ord(character) <= 0xDFFF for character in value
    ):
        _fail("JSON string contains an unpaired surrogate")
    if type(value) in {list, tuple}:
        for item in value:
            _reject_nonfinite(item)
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail("JSON object keys must be strings")
            _reject_nonfinite(key)
            _reject_nonfinite(item)


def _canonical_json_bytes(value: Any) -> bytes:
    _reject_nonfinite(value)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        _fail(f"model is not canonical JSON: {exc}")
    return text.encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    _reject_nonfinite(value)
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        _fail(f"model is not JSON: {exc}")
    return (text + "\n").encode("utf-8")


def _duplicate_rejecting_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    _fail(f"non-finite JSON number: {token}")


def _strict_json_value(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail(f"{label} is not UTF-8: {exc}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_constant,
        )
    except AncestryAuditError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        _fail(f"invalid {label}: {exc}")
    try:
        _reject_nonfinite(value)
    except RecursionError:
        _fail(f"invalid {label}: nesting exceeds safe limits")
    return value


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    value = _strict_json_value(payload, label)
    if type(value) is not dict:
        _fail(f"{label} is not an object")
    return value


def _unicode_fold(text: str, *, separated_marks: bool = False) -> str:
    folded: list[str] = []
    for character in unicodedata.normalize("NFKC", text).casefold():
        category = unicodedata.category(character)
        if category in {"Cf", "Me", "Mn"}:
            if separated_marks:
                folded.append(" ")
            continue
        if category == "Pd":
            folded.append("-")
        elif character == "×":
            folded.append("x")
        else:
            folded.append(character)
    return "".join(folded)


def _normalized_tokens(text: str) -> str:
    normalized: list[str] = []
    for character in _unicode_fold(text, separated_marks=True):
        category = unicodedata.category(character)
        if (
            character.isspace()
            or character in {"_", "/", "\\"}
            or (category.startswith("P") and character not in {".", "%"})
        ):
            normalized.append(" ")
        else:
            normalized.append(character)
    return " ".join("".join(normalized).split())


def _json_scalar_texts(value: Any) -> Iterable[str]:
    if type(value) is dict:
        for key, item in value.items():
            if key not in _JSON_FIXED_POLICY_KEYS:
                yield key
            yield from _json_scalar_texts(item)
    elif type(value) is list:
        for item in value:
            yield from _json_scalar_texts(item)
    elif type(value) is str:
        yield value
    elif type(value) in {int, float}:
        yield str(value)


def _json_value_texts(value: Any) -> Iterable[str]:
    if type(value) is dict:
        for item in value.values():
            yield from _json_value_texts(item)
    elif type(value) is list:
        for item in value:
            yield from _json_value_texts(item)
    elif type(value) is str:
        yield value
    elif type(value) in {int, float}:
        yield str(value)


def _design_input_texts(
    payload: bytes, media_type: str, identity_id: str
) -> Iterable[str]:
    if media_type not in _TEXT_MEDIA_TYPES:
        _fail(f"design input {identity_id} has unsupported media type")
    if not payload or len(payload) > MAX_DESIGN_INPUT_BYTES or b"\0" in payload:
        _fail(f"design input {identity_id} is empty, oversized, or contains NUL")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        _fail(f"design input {identity_id} is not canonical UTF-8 text")
    if media_type == "application/json":
        value = _strict_json_value(payload, f"design input {identity_id}")
        return _json_scalar_texts(value)
    return (text,)


def _forbidden_fragments() -> tuple[str, ...]:
    raw_fragments = list(_FORBIDDEN_RAW_FRAGMENTS)
    for quarantine_id, (path, _digest) in QUARANTINED_ANCESTRY.items():
        raw_fragments.extend((quarantine_id, path, path.rsplit("/", 1)[-1]))
    return tuple(
        sorted({_normalized_tokens(fragment) for fragment in raw_fragments})
    )


def _scan_texts(
    texts: Iterable[str], label: str, forbidden_fragments: tuple[str, ...]
) -> None:
    for text in texts:
        identifier_text = _unicode_fold(text)
        token_text = _normalized_tokens(text)
        if any(fragment in token_text for fragment in forbidden_fragments):
            _fail(f"{label} contains a quarantined normalized fragment")
        if (
            _QUARANTINE_ID_TOKEN.search(identifier_text)
            or _QUARANTINE_ID_TOKEN.search(token_text)
        ):
            _fail(f"{label} contains a quarantined normalized identifier")
        if (
            _NCU_TOKEN.search(identifier_text)
            or _NCU_TOKEN.search(token_text)
            or _METRIC_NAMESPACE.search(identifier_text)
        ):
            _fail(f"{label} contains a target-metric token")


def _variable_readout_projection(readout: Mapping[str, Any]) -> dict[str, Any]:
    method_contract = readout["method_contract"]
    return {
        "readout_id": readout["readout_id"],
        "method_class": readout["method_class"],
        "instrument": readout["instrument"],
        "derivation_identity_id": readout["derivation_identity_id"],
        "derivation_operation": readout["derivation_operation"],
        "method_contract": {
            "mechanism": method_contract["mechanism"],
            "estimator_identity": method_contract["estimator_identity"],
            "confounder_set": method_contract["confounder_set"],
            "control_evidence_roles": method_contract["control_evidence_roles"],
        },
        "required_evidence_roles": readout["required_evidence_roles"],
    }


def _variable_quantity_projection(quantity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "quantity_id": quantity["quantity_id"],
        "quantity_group": quantity["quantity_group"],
        "producer": quantity["producer"],
        "upstream_family_ids": quantity["upstream_family_ids"],
        "consumer_binding": quantity["consumer_binding"],
        "meaning": quantity["meaning"],
        "conditioning": quantity["conditioning"],
        "unit": quantity["unit"],
        "value_domain": quantity["value_domain"],
        "agreement_limit_fraction": quantity["agreement_limit_fraction"],
        "readout_contracts": [
            _variable_readout_projection(readout)
            for readout in quantity["readout_contracts"]
        ],
        "authorized_gap_kinds": quantity["authorized_gap_kinds"],
    }


def _scientific_projection(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "execution_order": authority["execution_order"],
        "scientific_rank": authority["scientific_rank"],
        "quantities": [
            _variable_quantity_projection(quantity)
            for quantity in authority["quantities"]
        ],
    }


def _quantity_contract_projection(
    quantity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "producer": quantity["producer"],
        "consumer_binding": quantity["consumer_binding"],
        "conditioning": quantity["conditioning"],
        "unit": quantity["unit"],
        "value_domain": quantity["value_domain"],
        "agreement_limit_fraction": quantity["agreement_limit_fraction"],
        "readout_contracts": [
            {
                key: value
                for key, value in readout.items()
                if key not in {"readout_id", "derivation_identity_id"}
            }
            for readout in quantity["readout_contracts"]
        ],
        "authorized_gap_kinds": quantity["authorized_gap_kinds"],
        "required": quantity["required"],
    }


def _dependency_edges(authority: Mapping[str, Any]) -> list[list[str]]:
    edges: set[tuple[str, str]] = set()
    for quantity in authority["quantities"]:
        producer = quantity["producer"]
        if producer["kind"] != "PROBE_FAMILY":
            continue
        producer_family_id = producer["family_id"]
        for upstream_family_id in quantity["upstream_family_ids"]:
            edges.add((upstream_family_id, producer_family_id))
    return [list(edge) for edge in sorted(edges)]


def _legacy_structural_fingerprints(
    legacy_authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_order_sha256": _sha256_bytes(
            _canonical_json_bytes(legacy_authority["execution_order"])
        ),
        "scientific_rank_sha256": _sha256_bytes(
            _canonical_json_bytes(legacy_authority["scientific_rank"])
        ),
        "quantity_contract_sha256": sorted(
            _sha256_bytes(
                _canonical_json_bytes(
                    _quantity_contract_projection(quantity)
                )
            )
            for quantity in legacy_authority["quantities"]
        ),
    }


def _validate_registry_entry(entry: Any) -> tuple[str, Mapping[str, Any]]:
    if type(entry) is not dict:
        _fail("source-surface registry entry is not an object")
    required = {
        "path",
        "category",
        "lifecycle",
        "authority_candidate",
        "can_collect",
        "reason",
    }
    if not required <= set(entry) or not set(entry) <= required | {
        "quarantine_id",
        "sha256",
    }:
        _fail("source-surface registry entry is not closed")
    path = entry["path"]
    reason = entry["reason"]
    if type(path) is not str or _SOURCE_PATH.fullmatch(path) is None:
        _fail("source-surface registry entry has an invalid path")
    if entry["category"] not in _SOURCE_CATEGORIES:
        _fail(f"source-surface registry category is invalid for {path}")
    if entry["lifecycle"] not in _SOURCE_LIFECYCLES:
        _fail(f"source-surface registry lifecycle is invalid for {path}")
    if entry["authority_candidate"] is not False or entry["can_collect"] is not False:
        _fail(f"source-surface registry grants authority or collection to {path}")
    if type(reason) is not str or not 1 <= len(reason) <= 256:
        _fail(f"source-surface registry reason is invalid for {path}")
    return path, entry


def _validate_registry(payload: bytes, schema_payload: bytes) -> str:
    registry = _strict_json(payload, "M5 source-surface registry")
    schema = _strict_json(schema_payload, "M5 source-surface registry schema")
    if payload != _pretty_json_bytes(registry):
        _fail("source-surface registry is not exact pretty JSON")
    if (
        schema.get("$id") != SOURCE_REGISTRY_SCHEMA_ID
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        _fail("source-surface registry schema contract changed")
    expected_root = {
        "schema_version": 1,
        "artifact_kind": "M5_SOURCE_SURFACE_REGISTRY",
        "authority_policy": "NO_SOURCE_SURFACE_IS_AN_M5_AUTHORITY",
        "card_write_policy": "FORBIDDEN",
        "collection_gate": "M5_COLLECT_DISABLED",
        "credit_state": "NO_REVIEW_OPERATOR_HARDWARE_OR_CAMPAIGN_CREDIT",
        "external_gate_state": "REVIEW_OPERATOR_HARDWARE_UNCHANGED",
        "milestone_state": "M5_NOT_STARTED",
        "closure_roots": list(_SOURCE_REGISTRY_CLOSURE_ROOTS),
    }
    if set(registry) != set(expected_root) | {"entries"} or any(
        registry.get(key) != value for key, value in expected_root.items()
    ):
        _fail("source-surface registry root contract changed")
    entries = registry["entries"]
    if type(entries) is not list or not 1 <= len(entries) <= 256:
        _fail("source-surface registry entries are absent or out of bounds")
    by_path: dict[str, Mapping[str, Any]] = {}
    for raw_entry in entries:
        path, entry = _validate_registry_entry(raw_entry)
        if path in by_path:
            _fail(f"source-surface registry duplicates path {path}")
        by_path[path] = entry
    for quarantine_id, (path, digest) in QUARANTINED_ANCESTRY.items():
        entry = by_path.get(path)
        if entry is None or entry.get("lifecycle") != "quarantined":
            _fail(f"source registry does not quarantine {quarantine_id}")
        if (
            entry.get("category") != "documentation"
            or entry.get("quarantine_id") != quarantine_id
            or entry.get("sha256") != digest
        ):
            _fail(f"source registry pin changed for {quarantine_id}")
    quarantined_paths = {
        path for path, _digest in QUARANTINED_ANCESTRY.values()
    }
    for path, entry in by_path.items():
        if path not in quarantined_paths and (
            "quarantine_id" in entry or "sha256" in entry
        ):
            _fail(f"source registry gives an unexpected quarantine pin to {path}")
        if _M5_RULING_PATH.fullmatch(path) and (
            entry["category"] != "documentation"
            or entry["lifecycle"] != "quarantined"
        ):
            _fail(f"source registry reclassifies quarantined ruling {path}")
        if _V1_CALIBRATION_SCHEMA_PATH.fullmatch(path) and (
            entry["category"] != "schema"
            or entry["lifecycle"] != "quarantined"
        ):
            _fail(f"source registry reclassifies quarantined v1 schema {path}")
        if _FORBIDDEN_SUBSTITUTE_PATH.fullmatch(path) and entry[
            "category"
        ] != "forbidden_substitute":
            _fail(f"source registry reclassifies forbidden substitute {path}")
    collector = by_path.get("tools/local_labels/m5_calibration_collector.py")
    if collector is None or (
        collector["category"], collector["lifecycle"]
    ) != ("collector", "disabled"):
        _fail("disabled M5 collector source classification changed")
    campaign_plan = by_path.get(
        "plans/sm86_comprehensive_live_capture_campaign_plan.md"
    )
    if campaign_plan is None or (
        campaign_plan["category"],
        campaign_plan["lifecycle"],
        campaign_plan["reason"],
    ) != (
        "documentation",
        "active",
        "Active scheduling plan; legacy design and rank references are not preregistration authority.",
    ):
        _fail("active campaign-plan source classification changed")
    return _sha256_bytes(payload)


def _validate_coverage(
    authority: Mapping[str, Any],
    design_inputs: Mapping[str, tuple[Mapping[str, Any], bytes]],
) -> str:
    provenance = authority.get("provenance")
    if type(provenance) is not dict:
        _fail("authority provenance is absent")
    bindings = provenance.get("design_input_bindings")
    sources = provenance.get("scientific_field_sources")
    if type(bindings) is not list or not bindings or type(sources) is not list:
        _fail("signed design-input bindings or scientific-field sources are absent")

    declared: dict[str, dict[str, Any]] = {}
    byte_bindings: set[tuple[str, int]] = set()
    for raw in bindings:
        if type(raw) is not dict or set(raw) != {
            "identity_id",
            "sha256",
            "size_bytes",
            "media_type",
            "role",
        }:
            _fail("design-input binding is not closed")
        identity_id = raw.get("identity_id")
        if type(identity_id) is not str or identity_id in declared:
            _fail("design-input identity is invalid or duplicated")
        byte_binding = (raw.get("sha256"), raw.get("size_bytes"))
        if byte_binding in byte_bindings:
            _fail("distinct design-input identities alias identical artifact bytes")
        byte_bindings.add(byte_binding)
        if raw.get("role") != "SCIENTIFIC_DESIGN_INPUT":
            _fail(f"design input {identity_id} has the wrong role")
        declared[identity_id] = raw
    if set(declared) != set(design_inputs):
        _fail("signed and retained design-input identity sets differ")
    artifact_ids: set[str] = set()
    artifact_paths: set[str] = set()
    for identity_id, binding in declared.items():
        artifact, payload = design_inputs[identity_id]
        expected = {
            "identity_id": identity_id,
            "sha256": artifact.get("sha256"),
            "size_bytes": artifact.get("size_bytes"),
            "media_type": artifact.get("media_type"),
            "role": "SCIENTIFIC_DESIGN_INPUT",
        }
        if binding != expected:
            _fail(f"signed binding is stale for design input {identity_id}")
        if (
            len(payload) != expected["size_bytes"]
            or _sha256_bytes(payload) != expected["sha256"]
        ):
            _fail(f"retained bytes differ from binding for design input {identity_id}")
        artifact_id = artifact.get("artifact_id")
        relative_path = artifact.get("relative_path")
        if artifact_id in artifact_ids or relative_path in artifact_paths:
            _fail("distinct design-input identities alias one retained artifact")
        artifact_ids.add(artifact_id)
        artifact_paths.add(relative_path)

    expected_pointers = {
        "/execution_order",
        "/scientific_rank",
        *(
            f"/quantities/{index}"
            for index in range(len(authority["quantities"]))
        ),
    }
    observed_pointers: set[str] = set()
    used_inputs: set[str] = set()
    for raw in sources:
        if type(raw) is not dict or set(raw) != {
            "field_pointer",
            "design_input_identity_ids",
        }:
            _fail("scientific-field source binding is not closed")
        pointer = raw.get("field_pointer")
        input_ids = raw.get("design_input_identity_ids")
        if type(pointer) is not str or pointer in observed_pointers:
            _fail("scientific-field source pointer is invalid or duplicated")
        if (
            type(input_ids) is not list
            or not input_ids
            or any(type(identity_id) is not str for identity_id in input_ids)
            or len(input_ids) != len(set(input_ids))
            or not set(input_ids) <= set(declared)
        ):
            _fail(f"scientific-field source binding is invalid for {pointer}")
        observed_pointers.add(pointer)
        used_inputs.update(input_ids)
    if observed_pointers != expected_pointers or used_inputs != set(declared):
        _fail("scientific-field coverage is incomplete or contains extra fields")
    return _sha256_bytes(_canonical_json_bytes(sources))


def build_audit(
    *,
    authority: Mapping[str, Any],
    authority_raw: bytes,
    design_inputs: Mapping[str, tuple[Mapping[str, Any], bytes]],
    source_registry_payload: bytes,
    source_registry_schema_payload: bytes,
    audit_schema_payload: bytes,
    legacy_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic audit proving only listed mechanical non-revival."""

    if not _is_sha256(_EXECUTING_TOOL_SHA256):
        _fail("retained audit-tool execution binding is absent")
    retained_authority = _strict_json(
        authority_raw, "M5 v2 quantity authority"
    )
    if (
        retained_authority != authority
        or authority_raw != _pretty_json_bytes(retained_authority)
    ):
        _fail("authority model differs from its exact retained bytes")
    source_registry_sha256 = _validate_registry(
        source_registry_payload, source_registry_schema_payload
    )
    field_sources_sha256 = _validate_coverage(authority, design_inputs)
    forbidden_fragments = _forbidden_fragments()
    quarantined_digests = {
        digest for _path, digest in QUARANTINED_ANCESTRY.values()
    }
    for identity_id, (artifact, payload) in sorted(design_inputs.items()):
        if artifact["sha256"] in quarantined_digests:
            _fail(
                f"design input {identity_id} exactly revives quarantined ancestry"
            )
        _scan_texts(
            _design_input_texts(
                payload, artifact["media_type"], identity_id
            ),
            f"design input {identity_id}",
            forbidden_fragments,
        )

    projection = _scientific_projection(authority)
    _scan_texts(
        _json_value_texts(projection),
        "variable scientific authority fields",
        forbidden_fragments,
    )

    legacy_fingerprints = _legacy_structural_fingerprints(legacy_authority)
    if authority["execution_order"] == legacy_authority["execution_order"]:
        _fail("authority revives the quarantined execution order")
    if authority["scientific_rank"] == legacy_authority["scientific_rank"]:
        _fail("authority revives the quarantined scientific rank")
    legacy_contracts = set(legacy_fingerprints["quantity_contract_sha256"])
    for quantity in authority["quantities"]:
        digest = _sha256_bytes(
            _canonical_json_bytes(_quantity_contract_projection(quantity))
        )
        if digest in legacy_contracts:
            _fail("authority revives a quarantined quantity contract")

    dependency_edges = _dependency_edges(authority)
    legacy_dependency_edges = [list(edge) for edge in LEGACY_DEPENDENCY_EDGES]
    revived_dependency_edges = sorted(
        {tuple(edge) for edge in dependency_edges}
        & set(LEGACY_DEPENDENCY_EDGES)
    )
    if revived_dependency_edges:
        _fail("authority revives quarantined dependency edges")

    quarantine_catalog_sha256 = _sha256_bytes(
        _canonical_json_bytes(QUARANTINED_ANCESTRY)
    )
    normalization_policy_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            {
                "normalization": _NORMALIZATION_POLICY,
                "forbidden_fragments": forbidden_fragments,
                "metric_namespace_pattern": _METRIC_NAMESPACE.pattern,
                "ncu_token_pattern": _NCU_TOKEN.pattern,
                "quarantine_id_pattern": _QUARANTINE_ID_TOKEN.pattern,
                "text_media_types": sorted(_TEXT_MEDIA_TYPES),
                "maximum_design_input_bytes": MAX_DESIGN_INPUT_BYTES,
            }
        )
    )
    legacy_fingerprints_sha256 = _sha256_bytes(
        _canonical_json_bytes(legacy_fingerprints)
    )
    legacy_dependency_edges_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            {
                "source_commit": LEGACY_DEPENDENCY_SOURCE_COMMIT,
                "edges": legacy_dependency_edges,
            }
        )
    )
    scientific_dependency_edges_sha256 = _sha256_bytes(
        _canonical_json_bytes(dependency_edges)
    )
    audit_tool_sha256 = _EXECUTING_TOOL_SHA256
    audit_schema_sha256 = _sha256_bytes(audit_schema_payload)
    source_registry_schema_sha256 = _sha256_bytes(
        source_registry_schema_payload
    )
    audit_policy_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            {
                "audit_policy_version": AUDIT_POLICY_VERSION,
                "audit_tool_sha256": audit_tool_sha256,
                "audit_schema_sha256": audit_schema_sha256,
                "source_registry_schema_sha256": (
                    source_registry_schema_sha256
                ),
                "quarantine_catalog_sha256": quarantine_catalog_sha256,
                "normalization_policy_sha256": normalization_policy_sha256,
                "legacy_structural_fingerprints_sha256": (
                    legacy_fingerprints_sha256
                ),
                "legacy_dependency_edges_sha256": (
                    legacy_dependency_edges_sha256
                ),
            }
        )
    )
    design_input_rows = [
        {"identity_id": identity_id, "artifact": dict(artifact)}
        for identity_id, (artifact, _payload) in sorted(
            design_inputs.items()
        )
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "M5_AUTHORITY_ANCESTRY_AUDIT",
        "audit_policy_version": AUDIT_POLICY_VERSION,
        "audit_policy_sha256": audit_policy_sha256,
        "audit_tool_sha256": audit_tool_sha256,
        "audit_schema_sha256": audit_schema_sha256,
        "authority_sha256": _sha256_bytes(authority_raw),
        "source_registry_sha256": source_registry_sha256,
        "source_registry_schema_sha256": source_registry_schema_sha256,
        "quarantine_catalog_sha256": quarantine_catalog_sha256,
        "normalization_policy_sha256": normalization_policy_sha256,
        "legacy_structural_fingerprints_sha256": (
            legacy_fingerprints_sha256
        ),
        "legacy_dependency_edges_sha256": legacy_dependency_edges_sha256,
        "scientific_dependency_edges_sha256": (
            scientific_dependency_edges_sha256
        ),
        "scientific_projection_sha256": _sha256_bytes(
            _canonical_json_bytes(projection)
        ),
        "scientific_field_sources_sha256": field_sources_sha256,
        "design_inputs": design_input_rows,
        "checks": {
            "declared_input_binding_complete": True,
            "scientific_field_coverage_complete": True,
            "source_registry_contract_validated": True,
            "retained_tool_execution_bound": True,
            "audit_policy_byte_bound": True,
            "exact_quarantine_digest_absent": True,
            "normalized_forbidden_tokens_absent": True,
            "variable_scientific_fields_scanned": True,
            "legacy_structural_fingerprints_absent": True,
            "legacy_dependency_edges_absent": True,
        },
        "verdict": "NO_LISTED_REVIVAL_DETECTED",
        "mechanical_non_revival_only": True,
        "semantic_review_required": True,
        "external_credit": "NONE",
    }
