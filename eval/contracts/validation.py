"""Production validation for the frozen P1 JSON contracts.

This module owns strict JSON decoding, Draft 2020-12 reference resolution,
canonical seals, trust-bound registry projection admission, and narrow semantic
contract guards. It deliberately does not join labels to predictions or compute
P2 accuracy scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jsonschema import ValidationError
from jsonschema.validators import validator_for
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_REGISTRY_SHA256 = (
    "95d90f8c4fffd9376b8d785047a9e02ecafc81ed43da6081985958c2b9dec62d"
)

METRIC_KEYS = (
    "smsp_inst_executed_sum",
    "traffic_gmem_ld_sectors",
    "traffic_gmem_st_sectors",
    "traffic_dram_read_sectors",
    "cache_hit_rate_l1tex_read_sector",
    "cache_hit_rate_l1tex_sector",
    "cache_hit_rate_l2_sector",
    "exec_cycles_gpc_max",
)
REPLAY_AUTHENTICATED_LAUNCH_FIELDS = frozenset(
    {
        "kernel_mangled_name",
        "grid",
        "block",
        "device_id",
        "stream_id",
        "binary_architecture",
    }
)
COHORT_ASSERTION_LAUNCH_FIELDS = frozenset(
    {"workload_id", "matched_execution_id", "process_launch_ordinal"}
)

_METRIC_KEY_SET = frozenset(METRIC_KEYS)
_CYCLES_KEY = "exec_cycles_gpc_max"
_CYCLES_BLOCKER = {
    "status": "required_unimplemented",
    "blocking": "aggregation_semantics_study",
}
_PROJECTED_ROW_FIELDS = (
    "key",
    "ncu_metric",
    "unit",
    "family",
    "metric_class",
    "carrier",
    "valid_min",
    "valid_max",
    "parser_transform",
    "aggregation",
    "repeatability_rule",
    "nullable",
    "deployability",
    "availability",
    "per_arch_availability",
    "model_io_role",
)

_SCHEMA_PATHS = {
    "launch_key": ROOT / "eval/schemas/launch_key.v1.schema.json",
    "manifest": ROOT / "sim/runner/schema/sim_run_manifest.v2.schema.json",
    "result": ROOT / "sim/runner/schema/sim_result.v2.schema.json",
    "sim_error": ROOT / "sim/runner/schema/sim_error.v2.schema.json",
    "matched_labels": ROOT / "eval/schemas/matched_labels.v1.schema.json",
    "projection": ROOT / "eval/schemas/metric_registry_projection.v1.schema.json",
    "report": ROOT / "eval/schemas/accuracy_report.v1.schema.json",
}
_PROJECTION_PATH = ROOT / "eval/contracts/metric_registry.7plus1.v1.json"
_REGISTRY_PATH = ROOT / "tools/local_labels/metric_registry.json"


class ContractValidationError(ValueError):
    """Typed, fail-closed P1 contract admission error."""

    def __init__(
        self, code: str, detail: str, *, context: Mapping[str, Any] | None = None
    ) -> None:
        self.code = code
        self.detail = detail
        self.context = dict(context or {})
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class MatchedLabelAuthority:
    """Identities supplied independently of report and matched-label documents."""

    matched_labels_sha256: str
    label_set_id: str
    label_set_sha256: str
    crosswalk_id: str
    crosswalk_sha256: str
    authenticated_empty_cohort_sha256: str | None = None


def _reject_json_constant(token: str) -> None:
    raise ContractValidationError(
        "JSON_CONSTANT_INVALID", f"non-finite JSON constant is forbidden: {token}"
    )


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ContractValidationError(
            "JSON_NUMBER_NONFINITE",
            "JSON number overflows the finite binary64 contract",
            context={"token": token},
        )
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ContractValidationError(
                "DUPLICATE_JSON_KEY",
                "JSON object repeats a key",
                context={"key": key},
            )
        value[key] = child
    return value


def loads_json_strict(text: str, *, source: str = "<memory>") -> Any:
    """Decode JSON with duplicate-key rejection and finite numeric values.

    Integer decoding intentionally retains the JSON integer exactly; schemas, not
    a legacy 2^53 guard, apply the uint64 bounds. Floating tokens that overflow
    binary64 are rejected during decoding rather than reaching a later schema or
    canonical-seal check as infinity.
    """

    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_int=int,
            parse_float=_parse_finite_float,
            parse_constant=_reject_json_constant,
        )
    except ContractValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise ContractValidationError(
            "JSON_DECODE_FAILED",
            exc.msg,
            context={"source": source, "line": exc.lineno, "column": exc.colno},
        ) from exc


def load_json_file_strict(path: str | Path) -> Any:
    source = Path(path)
    return loads_json_strict(source.read_text(encoding="utf-8"), source=str(source))


def _read_schema(path: Path) -> dict[str, Any]:
    value = load_json_file_strict(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"schema root is not an object: {path}")
    return value


_SCHEMAS = {name: _read_schema(path) for name, path in _SCHEMA_PATHS.items()}
_SCHEMA_REGISTRY = Registry()
for _schema in _SCHEMAS.values():
    _SCHEMA_REGISTRY = _SCHEMA_REGISTRY.with_resource(
        _schema["$id"], Resource.from_contents(_schema)
    )

_VALIDATORS = {}
for _name, _schema in _SCHEMAS.items():
    _validator_class = validator_for(_schema)
    _validator_class.check_schema(_schema)
    _VALIDATORS[_name] = _validator_class(_schema, registry=_SCHEMA_REGISTRY)


def schema_document(name: str) -> dict[str, Any]:
    try:
        return deepcopy(_SCHEMAS[name])
    except KeyError as exc:
        raise KeyError(f"unknown P1 contract {name!r}") from exc


def canonical_sha256(document: Mapping[str, Any], omitted_member: str) -> str:
    unsigned = deepcopy(dict(document))
    unsigned.pop(omitted_member, None)
    try:
        canonical = json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            "CANONICALIZATION_FAILED", str(exc), context={"member": omitted_member}
        ) from exc
    return hashlib.sha256(canonical).hexdigest()


def _json_path(error: ValidationError) -> str:
    path = "$"
    for component in error.absolute_path:
        path += f"[{component}]" if isinstance(component, int) else f".{component}"
    return path


def _validate_schema(name: str, document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ContractValidationError(
            "SCHEMA_VALIDATION_FAILED",
            f"{name} document must be an object",
            context={"contract": name, "json_path": "$"},
        )
    try:
        _VALIDATORS[name].validate(document)
    except ValidationError as exc:
        raise ContractValidationError(
            "SCHEMA_VALIDATION_FAILED",
            exc.message,
            context={
                "contract": name,
                "json_path": _json_path(exc),
                "validator": str(exc.validator),
            },
        ) from exc
    return document


def _validate_seal(
    document: Mapping[str, Any], seal_member: str, *, contract: str
) -> None:
    expected = canonical_sha256(document, seal_member)
    observed = document[seal_member]
    if observed != expected:
        raise ContractValidationError(
            "CANONICAL_SEAL_MISMATCH",
            f"{contract} canonical SHA-256 seal does not match",
            context={
                "contract": contract,
                "seal_member": seal_member,
                "expected_sha256": expected,
                "observed_sha256": observed,
            },
        )


def validate_registry_projection() -> dict[str, Any]:
    projection = _validate_schema("projection", load_json_file_strict(_PROJECTION_PATH))
    _validate_seal(
        projection, "projection_sha256", contract="metric_registry_projection.v1"
    )

    source_bytes = _REGISTRY_PATH.read_bytes()
    observed_source_hash = hashlib.sha256(source_bytes).hexdigest()
    if observed_source_hash != CANONICAL_REGISTRY_SHA256:
        raise ContractValidationError(
            "REGISTRY_SOURCE_HASH_MISMATCH",
            "canonical registry bytes do not match the P1 trust pin",
            context={
                "expected_sha256": CANONICAL_REGISTRY_SHA256,
                "observed_sha256": observed_source_hash,
            },
        )
    if projection["source_registry_sha256"] != observed_source_hash:
        raise ContractValidationError(
            "REGISTRY_PROJECTION_SOURCE_MISMATCH",
            "projection source hash does not bind the canonical registry bytes",
        )

    source = loads_json_strict(source_bytes.decode("utf-8"), source=str(_REGISTRY_PATH))
    source_rows = {row["key"]: row for row in source["metrics"]}
    expected_rows = [
        {field: source_rows[key][field] for field in _PROJECTED_ROW_FIELDS}
        for key in METRIC_KEYS
    ]
    expected_envelope = {
        "source_registry_version": source["registry_version"],
        "source_label_schema_version": source["label_schema_version"],
        "repeatability_thresholds": source["repeatability_thresholds"],
        "metrics": expected_rows,
    }
    for field, expected in expected_envelope.items():
        if projection[field] != expected:
            raise ContractValidationError(
                "REGISTRY_PROJECTION_CONTENT_MISMATCH",
                "retained projection is not the exact canonical-source projection",
                context={"field": field},
            )
    return projection


def validate_schema_definitions() -> None:
    for schema in _SCHEMAS.values():
        validator_for(schema).check_schema(schema)
    validate_registry_projection()


def _launch_token_unchecked(key: Mapping[str, Any]) -> str:
    normalized = {
        "workload_id": key["workload_id"],
        "matched_execution_id": key["matched_execution_id"],
        "process_launch_ordinal": int(key["process_launch_ordinal"]),
        "kernel_mangled_name": key["kernel_mangled_name"],
        "grid": {axis: int(key["grid"][axis]) for axis in ("x", "y", "z")},
        "block": {axis: int(key["block"][axis]) for axis in ("x", "y", "z")},
        "device_id": int(key["device_id"]),
        "stream_id": int(key["stream_id"]),
        "binary_architecture": key["binary_architecture"],
    }
    return json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def canonical_launch_key_token(key: Any) -> str:
    """Validate and canonically tokenize the exact composite launch key."""

    validated = validate_launch_key(key)
    return _launch_token_unchecked(validated)


def _unique_launch_tokens(
    launches: list[dict[str, Any]], *, contract: str
) -> list[str]:
    tokens: list[str] = []
    first_index: dict[str, int] = {}
    for index, launch in enumerate(launches):
        token = _launch_token_unchecked(launch["launch_key"])
        if token in first_index:
            raise ContractValidationError(
                "DUPLICATE_LAUNCH_KEY",
                f"{contract} repeats an exact launch key",
                context={
                    "contract": contract,
                    "first_index": first_index[token],
                    "duplicate_index": index,
                },
            )
        first_index[token] = index
        tokens.append(token)
    return tokens


def _validate_claimability(
    dirty: bool, claimability: Mapping[str, Any], *, identity: str
) -> None:
    expected = "unclaimable" if dirty else "claimable"
    if claimability["status"] != expected:
        raise ContractValidationError(
            "CLAIMABILITY_MISMATCH",
            f"{identity} dirty state contradicts claimability",
            context={"identity": identity, "dirty": dirty, "expected": expected},
        )


def validate_launch_key(document: Any) -> dict[str, Any]:
    return _validate_schema("launch_key", document)


def validate_manifest(document: Any) -> dict[str, Any]:
    manifest = _validate_schema("manifest", document)
    _unique_launch_tokens(manifest["launches"], contract="manifest")
    return manifest


def _validate_fraction(outcome: Mapping[str, Any], *, location: str) -> None:
    if outcome.get("status") != "predicted":
        return
    value = outcome.get("value")
    if not isinstance(value, dict) or value.get("kind") != "fraction":
        return
    numerator = value["numerator"]
    denominator = value["denominator"]
    observed = value["value"]
    if numerator > denominator:
        raise ContractValidationError(
            "FRACTION_INCONSISTENT",
            "fraction numerator exceeds denominator",
            context={"location": location},
        )
    if denominator == 0:
        if numerator != 0 or observed is not None:
            raise ContractValidationError(
                "FRACTION_INCONSISTENT",
                "zero denominator requires zero numerator and null value",
                context={"location": location},
            )
        return
    expected = numerator / denominator
    if observed != expected:
        raise ContractValidationError(
            "FRACTION_INCONSISTENT",
            "stored fraction value contradicts numerator and denominator",
            context={
                "location": location,
                "expected_value": expected,
                "observed_value": observed,
            },
        )


def _validate_metric_map(metrics: Mapping[str, Any], *, location: str) -> None:
    if frozenset(metrics) != _METRIC_KEY_SET:
        raise ContractValidationError(
            "METRIC_SET_MISMATCH",
            "metric map is not the exact retained 7+1 set",
            context={"location": location},
        )
    if metrics[_CYCLES_KEY] != _CYCLES_BLOCKER:
        raise ContractValidationError(
            "CYCLES_BLOCKER_MISMATCH",
            "cycles must remain RequiredUnimplemented(AggregationSemanticsStudy)",
            context={"location": location},
        )
    for key, outcome in metrics.items():
        _validate_fraction(outcome, location=f"{location}.{key}")


def _expected_unavailable(needs: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "detail": {"reason": "mem_mode_lacks_unit", "needs": needs},
    }


def _validate_replay_availability(result: Mapping[str, Any]) -> None:
    mode = result["resolved_configuration"]["replay_mode"]["mode"]
    expected_status: dict[str, str] = {"smsp_inst_executed_sum": "predicted"}
    l1_keys = {
        "traffic_gmem_ld_sectors",
        "traffic_gmem_st_sectors",
        "cache_hit_rate_l1tex_read_sector",
        "cache_hit_rate_l1tex_sector",
    }
    if mode == "flat":
        expected_status.update({key: "unavailable" for key in l1_keys})
        expected_status["cache_hit_rate_l2_sector"] = "unavailable"
        expected_status["traffic_dram_read_sectors"] = "unavailable"
    elif mode == "l1_const":
        expected_status.update({key: "predicted" for key in l1_keys})
        expected_status["cache_hit_rate_l2_sector"] = "unavailable"
        expected_status["traffic_dram_read_sectors"] = "unavailable"
    else:
        expected_status.update({key: "predicted" for key in l1_keys})
        expected_status["cache_hit_rate_l2_sector"] = "predicted"
        expected_status["traffic_dram_read_sectors"] = "predicted"

    maps = [launch["metrics"] for launch in result["launches"]]
    maps.append(result["aggregate"]["metrics"])
    for map_index, metrics in enumerate(maps):
        for key, status in expected_status.items():
            outcome = metrics[key]
            if outcome["status"] != status:
                raise ContractValidationError(
                    "REPLAY_AVAILABILITY_MISMATCH",
                    "metric availability contradicts resolved replay configuration",
                    context={"map_index": map_index, "metric_key": key, "mode": mode},
                )
            if status == "unavailable":
                needs = (
                    "l1tex"
                    if key in l1_keys
                    else "lts"
                    if key == "cache_hit_rate_l2_sector"
                    else "dram"
                )
                if outcome != _expected_unavailable(needs):
                    raise ContractValidationError(
                        "REPLAY_AVAILABILITY_MISMATCH",
                        "typed unavailable detail names the wrong modeled unit",
                        context={"map_index": map_index, "metric_key": key},
                    )


def _derive_aggregate_outcome(key: str, outcomes: list[Mapping[str, Any]]) -> Any:
    if key == _CYCLES_KEY:
        return _CYCLES_BLOCKER
    if all(outcome == outcomes[0] for outcome in outcomes) and outcomes[0]["status"] != "predicted":
        return deepcopy(outcomes[0])
    if all(
        outcome.get("status") == "predicted"
        and outcome["value"]["kind"] == "exact"
        for outcome in outcomes
    ):
        return {
            "status": "predicted",
            "value": {
                "kind": "exact",
                "semantics": "simulator_scope",
                "value": sum(outcome["value"]["value"] for outcome in outcomes),
            },
        }
    if all(
        outcome.get("status") == "predicted"
        and outcome["value"]["kind"] == "fraction"
        for outcome in outcomes
    ):
        numerator = sum(outcome["value"]["numerator"] for outcome in outcomes)
        denominator = sum(outcome["value"]["denominator"] for outcome in outcomes)
        return {
            "status": "predicted",
            "value": {
                "kind": "fraction",
                "semantics": "simulator_scope",
                "value": None if denominator == 0 else numerator / denominator,
                "numerator": numerator,
                "denominator": denominator,
            },
        }
    raise ContractValidationError(
        "AGGREGATE_METRIC_INCONSISTENT",
        "per-launch metric outcomes have no frozen aggregate composition",
        context={"metric_key": key},
    )


def _validate_aggregate_metrics(result: Mapping[str, Any]) -> None:
    aggregate = result["aggregate"]["metrics"]
    for key in METRIC_KEYS:
        expected = _derive_aggregate_outcome(
            key, [launch["metrics"][key] for launch in result["launches"]]
        )
        if aggregate[key] != expected:
            raise ContractValidationError(
                "AGGREGATE_METRIC_INCONSISTENT",
                "aggregate metric does not compose from per-launch outcomes",
                context={"metric_key": key},
            )


def _validate_trace_bundle(trace: Mapping[str, Any]) -> None:
    for field in ("total_size_bytes", "record_count", "instruction_count"):
        if trace[field] <= 0:
            raise ContractValidationError(
                "VACUOUS_SUCCESS_TRACE",
                f"successful result has nonpositive trace {field}",
                context={"field": field, "value": trace[field]},
            )
    components = trace["components"]
    component_bytes = (
        components["root_dynamic_trace"]["size_bytes"]
        + components["threadblock_payloads"]["total_size_bytes"]
        + components["execution_metadata"]["size_bytes"]
        + components["control_bits_sidecar"]["size_bytes"]
        + components["stats"]["size_bytes"]
    )
    if component_bytes != trace["total_size_bytes"]:
        raise ContractValidationError(
            "TRACE_BUNDLE_SIZE_MISMATCH",
            "trace bundle total bytes do not equal replay-consumed component bytes",
            context={"component_bytes": component_bytes, "total_size_bytes": trace["total_size_bytes"]},
        )


def validate_result(document: Any) -> dict[str, Any]:
    result = _validate_schema("result", document)
    _validate_seal(result, "result_sha256", contract="sim_result.v2")
    _validate_trace_bundle(result["trace_identity"])
    _validate_claimability(
        result["build_identity"]["sim_git_dirty"],
        result["build_identity"]["claimability"],
        identity="simulator_build",
    )
    result_tokens = _unique_launch_tokens(result["launches"], contract="result")
    diagnostic_tokens = _unique_launch_tokens(
        result["diagnostics"]["per_launch"], contract="result_diagnostics"
    )
    if set(result_tokens) != set(diagnostic_tokens):
        raise ContractValidationError(
            "DIAGNOSTIC_LAUNCH_SET_MISMATCH",
            "diagnostics launch keys do not equal the result launch set",
        )
    if result["aggregate"]["launch_count"] != len(result["launches"]):
        raise ContractValidationError(
            "AGGREGATE_LAUNCH_COUNT_MISMATCH",
            "aggregate launch_count does not equal launches length",
        )
    for index, launch in enumerate(result["launches"]):
        _validate_metric_map(launch["metrics"], location=f"launches[{index}].metrics")
    _validate_metric_map(result["aggregate"]["metrics"], location="aggregate.metrics")
    _validate_replay_availability(result)
    _validate_aggregate_metrics(result)
    return result


def validate_sim_error(document: Any) -> dict[str, Any]:
    error = _validate_schema("sim_error", document)
    _validate_seal(error, "error_sha256", contract="sim_error.v2")
    build = error.get("build_identity")
    if build is not None:
        _validate_claimability(
            build["sim_git_dirty"], build["claimability"], identity="simulator_build"
        )
    return error


def _validate_matched_authority(
    labels: Mapping[str, Any], authority: MatchedLabelAuthority
) -> None:
    observed = {
        "matched_labels_sha256": labels["matched_labels_sha256"],
        "label_set_id": labels["source_label_identity"]["label_set_id"],
        "label_set_sha256": labels["source_label_identity"]["label_set_sha256"],
        "crosswalk_id": labels["crosswalk_identity"]["crosswalk_id"],
        "crosswalk_sha256": labels["crosswalk_identity"]["crosswalk_sha256"],
    }
    expected = {
        "matched_labels_sha256": authority.matched_labels_sha256,
        "label_set_id": authority.label_set_id,
        "label_set_sha256": authority.label_set_sha256,
        "crosswalk_id": authority.crosswalk_id,
        "crosswalk_sha256": authority.crosswalk_sha256,
    }
    if observed != expected:
        raise ContractValidationError(
            "MATCHED_LABEL_AUTHORITY_MISMATCH",
            "matched-label document does not match independent authority hashes",
            context={"observed": observed, "expected": expected},
        )


def validate_matched_labels(
    document: Any, *, authority: MatchedLabelAuthority | None = None
) -> dict[str, Any]:
    if document is None:
        raise ContractValidationError(
            "LABEL_INPUT_UNAVAILABLE",
            "no independently admitted matched-label input was provided",
            context={"availability": "unavailable"},
        )
    if authority is None:
        raise ContractValidationError(
            "MATCHED_LABEL_AUTHORITY_UNAVAILABLE",
            "matched-label content cannot self-authorize its source and crosswalk hashes",
        )
    validate_registry_projection()
    labels = _validate_schema("matched_labels", document)
    _validate_seal(labels, "matched_labels_sha256", contract="matched_labels.v1")
    _validate_matched_authority(labels, authority)
    _unique_launch_tokens(labels["launches"], contract="matched_labels")
    for launch_index, launch in enumerate(labels["launches"]):
        seen: dict[str, int] = {}
        for metric_index, metric in enumerate(launch["metrics"]):
            key = metric["metric_key"]
            if key in seen:
                raise ContractValidationError(
                    "DUPLICATE_LABEL_METRIC",
                    "matched-label launch repeats a metric row",
                    context={"launch_index": launch_index, "metric_key": key},
                )
            seen[key] = metric_index
        if frozenset(seen) != _METRIC_KEY_SET:
            raise ContractValidationError(
                "LABEL_METRIC_SET_MISMATCH",
                "matched-label launch is not the exact retained 7+1 set",
                context={"launch_index": launch_index},
            )
    return labels


def _sum_unavailable(rows: list[Mapping[str, Any]]) -> int:
    return sum(int(row["count"]) for row in rows)


def _unavailable_discriminator(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["status"],
        row.get("reason"),
        row.get("needs"),
        row.get("detail"),
        row.get("blocking"),
    )


def _unavailable_counter(rows: list[Mapping[str, Any]]) -> Counter[tuple[Any, ...]]:
    counter: Counter[tuple[Any, ...]] = Counter()
    for row in rows:
        key = _unavailable_discriminator(row)
        if key in counter:
            raise ContractValidationError(
                "DUPLICATE_UNAVAILABLE_TYPE",
                "unavailable-by-type discriminator is duplicated",
            )
        counter[key] = int(row["count"])
    return counter


def _validate_error_stats_counts(
    stats: Mapping[str, Any], *, maximum_n: int, exact_n: int | None
) -> None:
    n = stats["n"]
    if n > maximum_n or (exact_n is not None and n != exact_n):
        raise ContractValidationError(
            "SUMMARY_COUNT_ALGEBRA_MISMATCH",
            "error-stat sample count contradicts the metric-family row",
        )
    values = (
        stats["mean_absolute_error"],
        stats["max_absolute_error"],
        stats["signed_bias"],
    )
    if n == 0 and any(value is not None for value in values):
        raise ContractValidationError(
            "SUMMARY_COUNT_ALGEBRA_MISMATCH",
            "zero-sample error statistics must be null",
        )
    if n > 0 and any(value is None for value in values):
        raise ContractValidationError(
            "SUMMARY_COUNT_ALGEBRA_MISMATCH",
            "nonempty error statistics cannot be null",
        )


def _validate_summary_counts(row: Mapping[str, Any]) -> None:
    summary = row["summary"]
    n_scored = row["n_scored"]
    if summary["kind"] == "count":
        if (
            summary["exact_count"] > n_scored
            or summary["zero_label_nonzero_prediction_count"] > n_scored
        ):
            raise ContractValidationError(
                "SUMMARY_COUNT_ALGEBRA_MISMATCH",
                "count-summary event count exceeds scored cells",
            )
        _validate_error_stats_counts(
            summary["absolute_error"], maximum_n=n_scored, exact_n=n_scored
        )
        _validate_error_stats_counts(
            summary["relative_error"], maximum_n=n_scored, exact_n=None
        )
    elif summary["kind"] == "rate":
        if summary["exact_count"] > n_scored:
            raise ContractValidationError(
                "SUMMARY_COUNT_ALGEBRA_MISMATCH",
                "rate-summary exact count exceeds scored cells",
            )
        _validate_error_stats_counts(
            summary["error_percentage_points"],
            maximum_n=n_scored,
            exact_n=n_scored,
        )


def _validate_noise_counts(row: Mapping[str, Any]) -> None:
    noise = row["within_label_noise"]
    if noise["evaluated_count"] + noise["not_applicable_count"] != row["n_scored"]:
        raise ContractValidationError(
            "WITHIN_LABEL_NOISE_ALGEBRA_MISMATCH",
            "evaluated and not-applicable noise cells must partition scored cells",
        )
    if noise["within_count"] > noise["evaluated_count"]:
        raise ContractValidationError(
            "WITHIN_LABEL_NOISE_ALGEBRA_MISMATCH",
            "within-noise count exceeds evaluated noise cells",
        )
    reasons = noise["not_applicable_reasons"]
    if (noise["not_applicable_count"] == 0) != (len(reasons) == 0):
        raise ContractValidationError(
            "WITHIN_LABEL_NOISE_ALGEBRA_MISMATCH",
            "not-applicable reasons must be present iff cells are not applicable",
        )


def _validate_nonnegative_absolute_errors(value: Any) -> None:
    nonnegative_fields = {
        "mean_absolute_error",
        "max_absolute_error",
        "mae_percent",
        "max_absolute_percentage_error",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key in nonnegative_fields and child is not None and child < 0:
                raise ContractValidationError(
                    "NEGATIVE_ABSOLUTE_ERROR",
                    "absolute-error statistic cannot be negative",
                )
            _validate_nonnegative_absolute_errors(child)
    elif isinstance(value, list):
        for child in value:
            _validate_nonnegative_absolute_errors(child)


def _validate_report_projection_identity(identities: Mapping[str, Any]) -> None:
    projection = validate_registry_projection()
    if identities["registry_projection_sha256"] != projection["projection_sha256"]:
        raise ContractValidationError(
            "REGISTRY_PROJECTION_IDENTITY_MISMATCH",
            "report does not bind the admitted exact projection seal",
        )


def _validate_report_label_binding(
    binding: Mapping[str, Any], authority: MatchedLabelAuthority
) -> None:
    expected = {
        "matched_labels_sha256": authority.matched_labels_sha256,
        "label_set_id": authority.label_set_id,
        "label_set_sha256": authority.label_set_sha256,
        "crosswalk_id": authority.crosswalk_id,
        "crosswalk_sha256": authority.crosswalk_sha256,
    }
    observed = {field: binding[field] for field in expected}
    if observed != expected:
        raise ContractValidationError(
            "REPORT_LABEL_BINDING_MISMATCH",
            "report label/crosswalk identities do not match independent authority",
        )


@dataclass(frozen=True)
class _RejectedFailurePolicy:
    failed_checks: frozenset[str]
    unknown_candidate_reason: str
    label_binding_status: str


_REJECTED_FAILURE_POLICIES = {
    "RESULT_SCHEMA_UNSUPPORTED": _RejectedFailurePolicy(
        frozenset({"result_schema"}), "result_unavailable", "bound"
    ),
    "RESULT_SCHEMA_INVALID": _RejectedFailurePolicy(
        frozenset({"result_schema"}), "result_unavailable", "bound"
    ),
    "RESULT_SEAL_INVALID": _RejectedFailurePolicy(
        frozenset({"result_seal"}), "result_unavailable", "bound"
    ),
    "LABEL_SCHEMA_UNSUPPORTED": _RejectedFailurePolicy(
        frozenset({"matched_label_schema"}), "label_input_unavailable", "unavailable"
    ),
    "LABEL_INPUT_UNAVAILABLE": _RejectedFailurePolicy(
        frozenset({"label_input_availability"}),
        "label_input_unavailable",
        "unavailable",
    ),
    "LABEL_SEAL_INVALID": _RejectedFailurePolicy(
        frozenset({"matched_label_seal"}), "label_input_unavailable", "unavailable"
    ),
    "CROSSWALK_IDENTITY_INVALID": _RejectedFailurePolicy(
        frozenset({"crosswalk_identity"}),
        "label_input_unavailable",
        "unavailable",
    ),
    "REGISTRY_PROJECTION_MISMATCH": _RejectedFailurePolicy(
        frozenset({"registry_projection"}),
        "admission_failed_before_enumeration",
        "bound",
    ),
    "ARCH_MISMATCH": _RejectedFailurePolicy(
        frozenset({"architecture"}),
        "admission_failed_before_enumeration",
        "bound",
    ),
    "UNIT_MISMATCH": _RejectedFailurePolicy(
        frozenset({"units"}), "admission_failed_before_enumeration", "bound"
    ),
}

_SCHEMA_REJECTED_FAILURE_TYPES = frozenset(
    branch["properties"]["failure_type"]["const"]
    for branch in _SCHEMAS["report"]["$defs"]["admissionFailure"]["oneOf"]
)
if frozenset(_REJECTED_FAILURE_POLICIES) != _SCHEMA_REJECTED_FAILURE_TYPES:
    raise RuntimeError("rejected failure policy is not exhaustive over report schema arms")


def _validate_authenticated_empty_cohort(
    candidate: Mapping[str, Any],
    binding: Mapping[str, Any],
    authority: MatchedLabelAuthority | None,
) -> None:
    if (
        binding["status"] != "bound"
        or authority is None
        or authority.authenticated_empty_cohort_sha256 is None
    ):
        raise ContractValidationError(
            "AUTHENTICATED_EMPTY_COHORT_AUTHORITY_UNAVAILABLE",
            "known-zero candidate status requires independent empty-cohort authority",
        )
    expected = {
        "empty_cohort_sha256": authority.authenticated_empty_cohort_sha256,
        "matched_labels_sha256": authority.matched_labels_sha256,
        "label_set_id": authority.label_set_id,
        "label_set_sha256": authority.label_set_sha256,
        "crosswalk_id": authority.crosswalk_id,
        "crosswalk_sha256": authority.crosswalk_sha256,
        "authority_role": "independently_authenticated_empty_cohort",
    }
    observed = candidate["authenticated_evidence"]
    if observed != expected:
        raise ContractValidationError(
            "AUTHENTICATED_EMPTY_COHORT_EVIDENCE_MISMATCH",
            "known-zero candidate evidence does not match independent authority",
            context={"observed": observed, "expected": expected},
        )


def _validate_unavailable_report_label_binding(
    binding: Mapping[str, Any], failures: list[Mapping[str, Any]]
) -> None:
    requested_label_set_id = binding["requested_label_set_id"]
    for failure in failures:
        context = failure["context"]
        context_label_set_id = context.get(
            "requested_label_set_id", context.get("label_set_id")
        )
        if (
            context_label_set_id is not None
            and context_label_set_id != requested_label_set_id
        ):
            raise ContractValidationError(
                "REPORT_LABEL_BINDING_MISMATCH",
                "rejected report label binding contradicts its failure context",
                context={"failure_type": failure["failure_type"]},
            )


def _validate_rejected_report(
    outcome: Mapping[str, Any],
    binding: Mapping[str, Any],
    authority: MatchedLabelAuthority | None,
) -> None:
    admission = outcome["admission"]
    checks = admission["checks"]
    failures = admission["failures"]
    failure_types = [failure["failure_type"] for failure in failures]
    if len(failure_types) != len(set(failure_types)):
        raise ContractValidationError(
            "DUPLICATE_ADMISSION_FAILURE_TYPE",
            "rejected admission repeats a typed failure arm",
        )

    policies = [_REJECTED_FAILURE_POLICIES[name] for name in failure_types]
    expected_failed_checks = set().union(
        *(policy.failed_checks for policy in policies)
    )
    observed_failed_checks = {
        name for name, status in checks.items() if status == "failed"
    }
    if observed_failed_checks != expected_failed_checks:
        raise ContractValidationError(
            "FAILURE_CONTEXT_MISMATCH",
            "rejected failure types do not exactly match failed check lanes",
            context={
                "expected_failed_checks": sorted(expected_failed_checks),
                "observed_failed_checks": sorted(observed_failed_checks),
            },
        )

    expected_binding_states = {policy.label_binding_status for policy in policies}
    if len(expected_binding_states) != 1 or binding["status"] not in expected_binding_states:
        raise ContractValidationError(
            "FAILURE_CONTEXT_MISMATCH",
            "rejected failure types contradict the report label-binding state",
            context={
                "expected_binding_states": sorted(expected_binding_states),
                "observed_binding_state": binding["status"],
            },
        )

    candidate = outcome["candidate_status"]
    if candidate["status"] == "known_zero":
        _validate_authenticated_empty_cohort(candidate, binding, authority)
    else:
        expected_reasons = {
            policy.unknown_candidate_reason for policy in policies
        }
        if len(expected_reasons) != 1 or candidate["reason"] not in expected_reasons:
            raise ContractValidationError(
                "FAILURE_CONTEXT_MISMATCH",
                "rejected failure types contradict candidate_status",
                context={
                    "expected_candidate_reasons": sorted(expected_reasons),
                    "observed_candidate_reason": candidate["reason"],
                },
            )

    if binding["status"] == "unavailable":
        _validate_unavailable_report_label_binding(binding, failures)

    if "LABEL_INPUT_UNAVAILABLE" in failure_types:
        for name in (
            "matched_label_schema",
            "matched_label_seal",
            "crosswalk_identity",
            "units",
        ):
            if checks[name] != "not_run":
                raise ContractValidationError(
                    "FAILURE_CONTEXT_MISMATCH",
                    "unavailable labels cannot be laundered through downstream checks",
                    context={"check": name},
                )


def _validate_admitted_report(outcome: Mapping[str, Any]) -> None:
    coverage = outcome["coverage"]
    if coverage["labeled_cells"] + coverage["unlabeled_cells"] != coverage[
        "total_cells"
    ]:
        raise ContractValidationError(
            "COVERAGE_ALGEBRA_MISMATCH",
            "labeled plus unlabeled cells must equal total cells",
        )
    if (
        coverage["scored_cells"]
        + coverage["unavailable_cells"]
        + coverage["comparison_failure_cells"]
        != coverage["total_cells"]
    ):
        raise ContractValidationError(
            "COVERAGE_ALGEBRA_MISMATCH",
            "scored, unavailable, and comparison failures must partition total",
        )
    if len(outcome["comparison_failures"]) != coverage["comparison_failure_cells"]:
        raise ContractValidationError(
            "COVERAGE_ALGEBRA_MISMATCH",
            "comparison failure records do not match coverage",
        )

    failed_cells: set[str] = set()
    for failure in outcome["comparison_failures"]:
        context = failure["context"]
        token = json.dumps(
            {
                "launch_key": context["launch_key"],
                "metric_key": context["metric_key"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if token in failed_cells:
            raise ContractValidationError(
                "DUPLICATE_COMPARISON_FAILURE",
                "a candidate launch-key/metric-key cell is counted more than once",
            )
        failed_cells.add(token)

    seen_rows: set[tuple[str, str]] = set()
    observed_keys: set[str] = set()
    labeled_sum = scored_sum = unavailable_sum = 0
    row_unavailable: Counter[tuple[Any, ...]] = Counter()
    for row in outcome["per_metric_family"]:
        row_key = (row["metric_key"], row["kernel_family"])
        if row_key in seen_rows:
            raise ContractValidationError(
                "DUPLICATE_METRIC_FAMILY_ROW",
                "metric-family summary row is duplicated",
            )
        seen_rows.add(row_key)
        observed_keys.add(row["metric_key"])
        labeled_sum += row["n_labeled"]
        scored_sum += row["n_scored"]
        unavailable_sum += row["n_unavailable"]
        if row["n_scored"] + row["n_unavailable"] > row["n_labeled"]:
            raise ContractValidationError(
                "ROW_COUNT_ALGEBRA_MISMATCH",
                "scored plus unavailable exceeds labeled cells",
            )
        row_counter = _unavailable_counter(row["unavailable_by_type"])
        if sum(row_counter.values()) != row["n_unavailable"]:
            raise ContractValidationError(
                "ROW_COUNT_ALGEBRA_MISMATCH",
                "row unavailable counts do not sum to n_unavailable",
            )
        row_unavailable.update(row_counter)
        _validate_summary_counts(row)
        _validate_noise_counts(row)
        if row["metric_key"] == _CYCLES_KEY:
            if row["n_scored"] != 0 or row["n_unavailable"] != row["n_labeled"]:
                raise ContractValidationError(
                    "CYCLES_REPORT_PROMOTION_FORBIDDEN",
                    "every admitted labeled cycle cell must remain unscored",
                )
            if row["summary"] != {
                "kind": "cycle",
                "status": "required_unimplemented",
                "blocking": "aggregation_semantics_study",
            }:
                raise ContractValidationError(
                    "CYCLES_REPORT_PROMOTION_FORBIDDEN",
                    "cycles summary is not the frozen required blocker",
                )

    if observed_keys != _METRIC_KEY_SET:
        raise ContractValidationError(
            "REPORT_METRIC_SET_MISMATCH",
            "admitted report must cover every retained 7+1 metric",
        )
    if labeled_sum != coverage["labeled_cells"]:
        raise ContractValidationError(
            "COVERAGE_ALGEBRA_MISMATCH", "n_labeled rows do not match coverage"
        )
    if scored_sum != coverage["scored_cells"]:
        raise ContractValidationError(
            "COVERAGE_ALGEBRA_MISMATCH", "n_scored rows do not match coverage"
        )
    if unavailable_sum != coverage["unavailable_cells"]:
        raise ContractValidationError(
            "COVERAGE_ALGEBRA_MISMATCH", "n_unavailable rows do not match coverage"
        )
    coverage_unavailable = _unavailable_counter(coverage["unavailable_by_type"])
    if coverage_unavailable != row_unavailable:
        raise ContractValidationError(
            "UNAVAILABLE_TYPE_ALGEBRA_MISMATCH",
            "coverage unavailable discriminators do not equal per-row totals",
        )
    _validate_nonnegative_absolute_errors(outcome)


def validate_report(
    document: Any, *, label_authority: MatchedLabelAuthority | None = None
) -> dict[str, Any]:
    report = _validate_schema("report", document)
    _validate_seal(report, "report_sha256", contract="accuracy_report.v1")
    identities = report["identities"]
    _validate_claimability(
        identities["simulator_git_dirty"],
        identities["simulator_build_claimability"],
        identity="simulator_build",
    )
    _validate_claimability(
        identities["evaluator_git_dirty"],
        identities["evaluator_build_claimability"],
        identity="evaluator_build",
    )
    _validate_report_projection_identity(identities)

    outcome = report["outcome"]
    binding = identities["label_binding"]
    if outcome["status"] == "rejected":
        _validate_rejected_report(outcome, binding, label_authority)
        if binding["status"] == "bound":
            if label_authority is None:
                raise ContractValidationError(
                    "MATCHED_LABEL_AUTHORITY_UNAVAILABLE",
                    "bound rejected report cannot self-authorize label/crosswalk identity",
                )
            _validate_report_label_binding(binding, label_authority)
        return report
    if label_authority is None:
        raise ContractValidationError(
            "MATCHED_LABEL_AUTHORITY_UNAVAILABLE",
            "admitted report cannot self-authorize matched-label/crosswalk identity",
        )
    _validate_report_label_binding(binding, label_authority)
    _validate_admitted_report(outcome)
    return report
