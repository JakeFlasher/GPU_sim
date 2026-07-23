#!/usr/bin/env python3
"""Build the 216-row minimum-footprint SOL representative trace census.

This additive authority keeps one coverage anchor for each of the 201
sm_86-applicable problems and 15 partial-native comparison companions.  It
selects only by the deterministic declared-tensor footprint rank, never by
observed latency.  Reduced anchors are non-native-predictive, and the 34
architecture-inapplicable problems remain typed, unscheduled inventory rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels import build_sol_workload_capture_matrix as full_builder  # noqa: E402


SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas/sol_representative_capture_matrix.schema.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/local_labels/campaign/sol_representative_capture_matrix.v1.json"
)
FULL_MATRIX_SCHEMA_PATH = full_builder.SCHEMA_PATH
INVENTORY_SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas/sol_live_workload_coverage_inventory.schema.json"
)
RUNNABILITY_SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas/sol_runnability.external.v2.schema.json"
)
SHRINK_SCHEMA_PATH = Path(
    "workloads/simbench_bw/schemas/sol_shrink_provenance_artifact.external.v3.schema.json"
)

EXPECTED_PROBLEMS = 235
EXPECTED_APPLICABLE_PROBLEMS = 201
EXPECTED_ARCH_INAPPLICABLE_PROBLEMS = 34
EXPECTED_ARCH_INAPPLICABLE_ROWS = 537
EXPECTED_NATIVE_SELECTIONS = 178
EXPECTED_PARTIAL_NATIVE_COMPANIONS = 15
EXPECTED_REDUCED_SELECTIONS = 23
ROWS_PER_CHUNK = 16
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ARCHITECTURE_REASONS = frozenset({
    "fp8_e4m3_contract",
    "nvfp4_e2m1_contract",
})
COHORT_ORDER = {"native": 0, "reduced_proxy": 1, "partial_native_row": 2}

_SHAPES_MODULE_PATH = (
    REPO_ROOT / "tools/sol_execbench/src/sol_execbench/core/data/shapes.py"
)
_SHAPES_SPEC = importlib.util.spec_from_file_location(
    "_sol_representative_shapes", _SHAPES_MODULE_PATH
)
if _SHAPES_SPEC is None or _SHAPES_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load SOL shape evaluator: {_SHAPES_MODULE_PATH}")
_SHAPES_MODULE = importlib.util.module_from_spec(_SHAPES_SPEC)
_SHAPES_SPEC.loader.exec_module(_SHAPES_MODULE)
resolve_shape_expression = _SHAPES_MODULE.resolve_shape_expression

DTYPE_BITS = {
    "float64": 64,
    "int64": 64,
    "float32": 32,
    "int32": 32,
    "float16": 16,
    "bfloat16": 16,
    "int16": 16,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "int8": 8,
    "bool": 8,
    "float4_e2m1": 4,
    "float4_e2m1fn_x2": 4,
}

REPRESENTATIVE_CAPTURE_CONTRACT = {
    "nvbit_trace": "full_unsampled",
    "ncu_profile": "full_unsampled_cycles_only_binding",
    "launch_scope": "all_in_profiler_roi",
    "dynamic_launch_scope": "all_inventory_launches",
    "partial_trace_allowed": False,
    "admission_required_layers": ["nvbit_trace", "ncu_profile"],
    "control_outcome_required": True,
    "control_outcome_policy": "complete_or_closed_typed_unavailable_nonblocking",
    "allowed_control_outcomes": [
        "complete",
        "typed_unavailable_no_stable_static_module",
    ],
    "allowed_typed_unavailable_reason": "origin_unresolved",
    "control_unavailability_invalidates_trace_ncu": False,
}


class RepresentativeMatrixError(ValueError):
    """The source authorities cannot support a closed representative plan."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepresentativeMatrixError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash the canonical artifact representation (including its trailing LF)."""
    return sha256_bytes(canonical_bytes(value))


def source_contract_sha256(value: Any) -> str:
    """Match the full-matrix producer's compact contract hash."""
    return full_builder.canonical_sha256(value)


def _read_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepresentativeMatrixError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RepresentativeMatrixError(f"{label} must be a JSON object: {path}")
    return value, payload


def _schema_validate(
    value: dict[str, Any], *, repo_root: Path, schema_path: Path, label: str,
) -> None:
    try:
        schema = json.loads((repo_root / schema_path).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(value, schema)
    except (OSError, UnicodeError, json.JSONDecodeError, jsonschema.ValidationError,
            jsonschema.SchemaError) as exc:
        raise RepresentativeMatrixError(f"{label} schema validation failed: {exc}") from exc


def _owner_input(value: Any, *, label: str) -> dict[str, str]:
    _require(isinstance(value, dict), f"{label} owner is absent")
    result = {
        "name": value.get("name"),
        "sha256": value.get("sha256"),
        "owner_seal_sha256": value.get("owner_seal_sha256"),
    }
    _require(
        isinstance(result["name"], str) and result["name"]
        and all(
            isinstance(result[key], str) and len(result[key]) == 64
            for key in ("sha256", "owner_seal_sha256")
        ),
        f"{label} owner identity is invalid",
    )
    return result


def _input_pin(
    *, name: str, payload: bytes, owner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "sha256": sha256_bytes(payload)}
    if owner is not None:
        result["owner"] = _owner_input(owner, label=name)
    return result


def _verify_source_matrix(matrix: dict[str, Any]) -> None:
    _require(
        matrix.get("schema_version") == 2
        and matrix.get("artifact_id") == "sm86_sol_workload_capture_matrix_live_v2",
        "source workload matrix identity is invalid",
    )
    seal = matrix.get("matrix_seal_sha256")
    _require(isinstance(seal, str) and len(seal) == 64,
             "source workload matrix seal is invalid")
    unsealed = {key: value for key, value in matrix.items()
                if key != "matrix_seal_sha256"}
    _require(
        full_builder.canonical_sha256(unsealed) == seal,
        "source workload matrix seal mismatch",
    )


def _index_variants(matrix: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    variants = matrix.get("variants")
    _require(isinstance(variants, list), "source matrix variants are absent")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in variants:
        _require(isinstance(variant, dict), "source matrix variant is malformed")
        key = (variant.get("slug"), variant.get("variant_id"))
        _require(
            all(isinstance(item, str) and item for item in key)
            and key not in result,
            f"source matrix variant identity is duplicated/invalid: {key!r}",
        )
        result[key] = variant
    return result


def _index_cells(document: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    cells = document.get("cells")
    _require(isinstance(cells, list), f"{label} cells are absent")
    result: dict[str, dict[str, Any]] = {}
    for cell in cells:
        _require(isinstance(cell, dict), f"{label} cell is malformed")
        slug = cell.get("slug")
        _require(
            isinstance(slug, str) and slug and slug not in result,
            f"{label} slug is duplicated/invalid: {slug!r}",
        )
        result[slug] = cell
    return result


def _bound_definition(
    *, variant: dict[str, Any], repo_root: Path, shrink_root: Path,
) -> tuple[dict[str, Any], str, str]:
    """Load the candidate's hash-pinned effective definition without fallback."""
    cohort = variant.get("cohort")
    if cohort == "reduced_proxy":
        provenance = variant.get("reduced_provenance")
        _require(isinstance(provenance, dict), "reduced definition provenance is absent")
        reference = provenance.get("reduced_definition_file")
        expected_sha = provenance.get("reduced_definition_file_sha256")
    else:
        reference = variant.get("definition_file")
        expected_sha = variant.get("definition_file_sha256")
    _require(
        isinstance(reference, str) and reference
        and isinstance(expected_sha, str) and len(expected_sha) == 64,
        f"effective definition binding is invalid: {variant.get('slug')}",
    )
    declared = Path(reference)
    _require(not declared.is_absolute() and ".." not in declared.parts,
             f"effective definition path is unsafe: {reference!r}")
    if len(declared.parts) == 1:
        candidate = shrink_root / declared
        _require(candidate.parent.resolve() == shrink_root.resolve(),
                 f"external definition leaves shrink root: {reference!r}")
    else:
        candidate = repo_root / declared
        try:
            candidate.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise RepresentativeMatrixError(
                f"effective definition leaves repository: {reference!r}"
            ) from exc
    _require(
        not candidate.is_symlink() and candidate.is_file(),
        f"effective definition is not a regular non-symlink file: {reference!r}",
    )
    payload = candidate.read_bytes()
    _require(sha256_bytes(payload) == expected_sha,
             f"effective definition hash mismatch: {reference!r}")
    try:
        definition = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RepresentativeMatrixError(
            f"effective definition is not valid JSON: {reference!r}: {exc}"
        ) from exc
    _require(isinstance(definition, dict),
             f"effective definition root is not an object: {reference!r}")
    return definition, reference, expected_sha


def definition_footprint(
    *, definition: dict[str, Any], capture_axes: dict[str, Any],
    definition_path: str, definition_sha256: str,
) -> dict[str, Any]:
    """Compute the exact definition-aware declared tensor-footprint rank."""
    axes = definition.get("axes")
    inputs = definition.get("inputs")
    outputs = definition.get("outputs")
    _require(isinstance(axes, dict), f"definition axes are absent: {definition_path}")
    _require(isinstance(inputs, dict), f"definition inputs are absent: {definition_path}")
    _require(isinstance(outputs, dict), f"definition outputs are absent: {definition_path}")
    _require(isinstance(capture_axes, dict), "capture axes are malformed")
    for name, value in capture_axes.items():
        _require(
            isinstance(name, str) and name
            and not isinstance(value, bool) and isinstance(value, int) and value >= 0,
            f"capture axis is invalid: {definition_path}:{name!r}={value!r}",
        )

    resolved: dict[str, int] = {}
    for name, specification in axes.items():
        _require(isinstance(name, str) and name and isinstance(specification, dict),
                 f"definition axis is malformed: {definition_path}:{name!r}")
        kind = specification.get("type")
        if kind == "const":
            value = specification.get("value")
            _require(not isinstance(value, bool) and isinstance(value, int) and value >= 0,
                     f"constant axis is invalid: {definition_path}:{name}")
        elif kind == "var":
            _require(name in capture_axes,
                     f"variable axis is unbound: {definition_path}:{name}")
            value = capture_axes[name]
        elif kind == "expr":
            expression = specification.get("expression")
            _require(isinstance(expression, str) and expression,
                     f"expression axis is invalid: {definition_path}:{name}")
            try:
                value = resolve_shape_expression(expression, resolved)
            except (TypeError, ValueError, NameError, SyntaxError, ZeroDivisionError,
                    OverflowError) as exc:
                raise RepresentativeMatrixError(
                    f"expression axis cannot resolve: {definition_path}:{name}: {exc}"
                ) from exc
            _require(not isinstance(value, bool) and isinstance(value, int) and value >= 0,
                     f"expression axis is negative/noninteger: {definition_path}:{name}")
        else:
            raise RepresentativeMatrixError(
                f"definition axis type is invalid: {definition_path}:{name}:{kind!r}"
            )
        resolved[name] = value
        if name in capture_axes:
            _require(capture_axes[name] == value,
                     f"capture axis disagrees with resolved axis: {definition_path}:{name}")

    extra_axes = sorted(set(capture_axes) - set(axes))
    scalar_controls: list[list[Any]] = []
    for name in extra_axes:
        specification = inputs.get(name)
        _require(
            isinstance(specification, dict) and specification.get("shape") is None,
            f"extra capture axis is not a declared scalar input: {definition_path}:{name}",
        )
        scalar_controls.append([name, capture_axes[name]])

    aggregate_bits = 0
    peak_bits = 0
    aggregate_elements = 0
    tensor_count = 0
    for surface_name, surface in (("inputs", inputs), ("outputs", outputs)):
        for tensor_name, specification in surface.items():
            _require(
                isinstance(tensor_name, str) and tensor_name
                and isinstance(specification, dict),
                f"tensor specification is malformed: {definition_path}:{surface_name}",
            )
            dtype = specification.get("dtype")
            _require(dtype in DTYPE_BITS,
                     f"tensor dtype is unsupported: {definition_path}:{tensor_name}:{dtype!r}")
            shape = specification.get("shape")
            if shape is None:
                elements = 1
            else:
                _require(isinstance(shape, list),
                         f"tensor shape is malformed: {definition_path}:{tensor_name}")
                dimensions: list[int] = []
                for dimension in shape:
                    _require(isinstance(dimension, str) and dimension,
                             f"tensor dimension is malformed: {definition_path}:{tensor_name}")
                    if dimension.isdigit():
                        value = int(dimension)
                    else:
                        _require(dimension in resolved,
                                 f"tensor dimension is unbound: {definition_path}:{tensor_name}:{dimension}")
                        value = resolved[dimension]
                    _require(value >= 0,
                             f"tensor dimension is negative: {definition_path}:{tensor_name}")
                    dimensions.append(value)
                elements = math.prod(dimensions)
            bits = elements * DTYPE_BITS[dtype]
            tensor_count += 1
            aggregate_elements += elements
            aggregate_bits += bits
            peak_bits = max(peak_bits, bits)
    _require(tensor_count > 0, f"definition declares no tensors: {definition_path}")
    axis_volume = math.prod(max(1, value) for value in resolved.values())
    return {
        "model": "declared_tensor_io_footprint_v1",
        "definition_path": definition_path,
        "definition_sha256": definition_sha256,
        "aggregate_declared_tensor_bits": aggregate_bits,
        "peak_declared_tensor_bits": peak_bits,
        "aggregate_declared_tensor_elements": aggregate_elements,
        "resolved_axis_volume": axis_volume,
        "scalar_control_tuple": scalar_controls,
        "tensor_count": tensor_count,
    }


def build_footprint_index(
    *, matrix: dict[str, Any], repo_root: Path, shrink_root: Path,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    variants = _index_variants(matrix)
    definition_cache: dict[tuple[str, str], dict[str, Any]] = {}
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in matrix.get("rows", []):
        _require(isinstance(row, dict) and isinstance(row.get("key"), dict),
                 "source matrix row is malformed during footprint construction")
        key = row["key"]
        identity = (key.get("slug"), key.get("source_workload_uuid"),
                    key.get("variant_id"))
        _require(identity not in result,
                 f"footprint row identity is duplicated: {identity!r}")
        variant = variants.get((identity[0], identity[2]))
        _require(isinstance(variant, dict),
                 f"footprint variant is absent: {identity!r}")
        definition, reference, expected_sha = _bound_definition(
            variant=variant, repo_root=repo_root, shrink_root=shrink_root,
        )
        definition_cache.setdefault((reference, expected_sha), definition)
        result[identity] = definition_footprint(
            definition=definition_cache[(reference, expected_sha)],
            capture_axes=row.get("capture_axes"),
            definition_path=reference,
            definition_sha256=expected_sha,
        )
    return result


def _positive_latency(diagnostic: dict[str, Any], *, identity: str) -> float:
    _require(
        diagnostic.get("diagnostic_present") is True
        and str(diagnostic.get("status")).lower() == "passed",
        f"selection diagnostic is not passing: {identity}",
    )
    trace = diagnostic.get("trace")
    evaluation = trace.get("evaluation") if isinstance(trace, dict) else None
    performance = evaluation.get("performance") if isinstance(evaluation, dict) else None
    latency = performance.get("latency_ms") if isinstance(performance, dict) else None
    _require(
        not isinstance(latency, bool)
        and isinstance(latency, (int, float))
        and math.isfinite(float(latency))
        and float(latency) > 0,
        f"selection diagnostic latency is invalid: {identity}",
    )
    return float(latency)


def _diagnostics_by_uuid(cell: dict[str, Any], *, slug: str) -> dict[str, dict[str, Any]]:
    diagnostics = cell.get("workload_diagnostics")
    _require(isinstance(diagnostics, list), f"diagnostics are absent: {slug}")
    result: dict[str, dict[str, Any]] = {}
    for diagnostic in diagnostics:
        _require(isinstance(diagnostic, dict), f"diagnostic is malformed: {slug}")
        uuid = diagnostic.get("uuid")
        _require(
            isinstance(uuid, str) and uuid and uuid not in result,
            f"diagnostic UUID is duplicated/invalid: {slug}:{uuid!r}",
        )
        result[uuid] = diagnostic
    return result


def _native_candidate(
    *, row: dict[str, Any], variant: dict[str, Any], run_cell: dict[str, Any],
    footprint: dict[str, Any],
) -> dict[str, Any]:
    key = row["key"]
    slug = key["slug"]
    uuid = key["source_workload_uuid"]
    _require(
        row.get("cohort") == "native"
        and row.get("claim_scope") == "native_workload_row"
        and row.get("native_shape") is True
        and row.get("native_predictive") is True
        and row.get("problem_level_runnable") is True
        and variant.get("cohort") == "native"
        and variant.get("native_shape") is True
        and variant.get("native_predictive") is True
        and run_cell.get("status") == "runnable",
        f"native selection contract is invalid: {slug}:{uuid}",
    )
    diagnostic = _diagnostics_by_uuid(run_cell, slug=slug).get(uuid)
    _require(isinstance(diagnostic, dict), f"native diagnostic is absent: {slug}:{uuid}")
    index = row.get("source_workload_index")
    _require(
        diagnostic.get("workload_index") == index
        and diagnostic.get("uuid") == uuid
        and diagnostic.get("axes") == row.get("source_axes")
        and diagnostic.get("workload_sha256") == row.get("source_row_sha256")
        and row.get("capture_workload_index") == index
        and row.get("capture_axes") == row.get("source_axes")
        and row.get("capture_row_sha256") == row.get("source_row_sha256")
        and row.get("row_evidence_sha256") == source_contract_sha256(diagnostic),
        f"native row/diagnostic binding mismatch: {slug}:{uuid}",
    )
    latency = _positive_latency(diagnostic, identity=f"{slug}:{uuid}")
    return _selection_record(
        row=row,
        variant=variant,
        selection_class="minimum_footprint_passing_native",
        selection_role="application_coverage_anchor",
        latency_ms=latency,
        evidence_sha256=source_contract_sha256(diagnostic),
        evidence_kind="runnability_workload_diagnostic",
        footprint=footprint,
    )


def _partial_native_candidate(
    *, row: dict[str, Any], variant: dict[str, Any], run_cell: dict[str, Any],
    footprint: dict[str, Any],
) -> dict[str, Any]:
    key = row["key"]
    slug = key["slug"]
    uuid = key["source_workload_uuid"]
    _require(
        row.get("cohort") == "partial_native_row"
        and row.get("claim_scope") == "partial_native_workload_row_only"
        and row.get("native_shape") is True
        and row.get("native_predictive") is False
        and row.get("problem_level_runnable") is False
        and variant.get("cohort") == "partial_native_row"
        and variant.get("native_shape") is True
        and variant.get("native_predictive") is False
        and run_cell.get("status") != "runnable",
        f"partial-native companion contract is invalid: {slug}:{uuid}",
    )
    diagnostic = _diagnostics_by_uuid(run_cell, slug=slug).get(uuid)
    _require(isinstance(diagnostic, dict), f"partial-native diagnostic is absent: {slug}:{uuid}")
    index = row.get("source_workload_index")
    _require(
        diagnostic.get("workload_index") == index
        and diagnostic.get("uuid") == uuid
        and diagnostic.get("axes") == row.get("source_axes")
        and diagnostic.get("workload_sha256") == row.get("source_row_sha256")
        and row.get("capture_workload_index") == index
        and row.get("capture_axes") == row.get("source_axes")
        and row.get("capture_row_sha256") == row.get("source_row_sha256")
        and row.get("row_evidence_sha256") == source_contract_sha256(diagnostic)
        and variant.get("selected_diagnostic_sha256")
        == source_contract_sha256(diagnostic),
        f"partial-native row/diagnostic binding mismatch: {slug}:{uuid}",
    )
    latency = _positive_latency(diagnostic, identity=f"{slug}:{uuid}:partial")
    return _selection_record(
        row=row,
        variant=variant,
        selection_class="minimum_footprint_passing_partial_native",
        selection_role="native_comparison_companion",
        latency_ms=latency,
        evidence_sha256=source_contract_sha256(diagnostic),
        evidence_kind="runnability_workload_diagnostic",
        footprint=footprint,
    )


def _reduced_candidate(
    *, row: dict[str, Any], variant: dict[str, Any], shrink_cell: dict[str, Any],
    footprint: dict[str, Any],
) -> dict[str, Any]:
    key = row["key"]
    slug = key["slug"]
    uuid = key["source_workload_uuid"]
    _require(
        row.get("cohort") == "reduced_proxy"
        and row.get("claim_scope") == "reduced_proxy_bundle_only"
        and row.get("native_shape") is False
        and row.get("native_predictive") is False
        and row.get("problem_level_runnable") is False
        and variant.get("cohort") == "reduced_proxy"
        and variant.get("native_shape") is False
        and variant.get("native_predictive") is False
        and shrink_cell.get("classification") == "reduced_representative"
        and shrink_cell.get("claim_scope") == "reduced_shape_only"
        and shrink_cell.get("native_predictive") is False
        and shrink_cell.get("reduced_run_status") == "runnable"
        and shrink_cell.get("correctness_result") == "passed",
        f"reduced-proxy selection contract is invalid: {slug}:{uuid}",
    )
    plans = shrink_cell.get("row_plans")
    _require(isinstance(plans, list), f"reduced row plans are absent: {slug}")
    matching_plans = [
        plan for plan in plans
        if isinstance(plan, dict)
        and plan.get("source_workload_uuid") == uuid
        and plan.get("variant_id") == key["variant_id"]
    ]
    _require(len(matching_plans) == 1, f"reduced row plan is not unique: {slug}:{uuid}")
    plan = matching_plans[0]
    _require(
        plan.get("planning_status") == "proxy_planned"
        and plan.get("native_predictive") is False
        and plan.get("claim_scope") == "reduced_shape_only"
        and plan.get("reduced_run_status") == "runnable"
        and plan.get("correctness_result") == "passed"
        and plan.get("source_workload_index") == row.get("source_workload_index")
        and plan.get("native_row_sha256") == row.get("source_row_sha256")
        and plan.get("reduced_row_sha256") == row.get("capture_row_sha256")
        and row.get("row_evidence_sha256") == source_contract_sha256(plan),
        f"reduced row/plan binding mismatch: {slug}:{uuid}",
    )
    provenance = variant.get("reduced_provenance")
    _require(
        isinstance(provenance, dict)
        and provenance.get("row_plan_sha256") == source_contract_sha256(plan),
        f"reduced variant plan binding mismatch: {slug}:{uuid}",
    )
    evidence = shrink_cell.get("cohort_execution_evidence")
    _require(isinstance(evidence, dict), f"reduced execution evidence is absent: {slug}")
    diagnostic = _diagnostics_by_uuid(evidence, slug=slug).get(uuid)
    _require(isinstance(diagnostic, dict), f"reduced diagnostic is absent: {slug}:{uuid}")
    _require(
        diagnostic.get("workload_index") == row.get("capture_workload_index")
        and diagnostic.get("uuid") == uuid
        and diagnostic.get("axes") == row.get("capture_axes")
        and diagnostic.get("workload_sha256") == row.get("capture_row_sha256"),
        f"reduced row/diagnostic binding mismatch: {slug}:{uuid}",
    )
    latency = _positive_latency(diagnostic, identity=f"{slug}:{uuid}:reduced")
    return _selection_record(
        row=row,
        variant=variant,
        selection_class="minimum_footprint_passing_reduced_proxy",
        selection_role="application_coverage_anchor",
        latency_ms=latency,
        evidence_sha256=source_contract_sha256(diagnostic),
        evidence_kind="shrink_cohort_workload_diagnostic",
        footprint=footprint,
    )


def _selection_record(
    *, row: dict[str, Any], variant: dict[str, Any], selection_class: str,
    selection_role: str, latency_ms: float, evidence_sha256: str,
    evidence_kind: str, footprint: dict[str, Any],
) -> dict[str, Any]:
    key = row["key"]
    scale_modified = selection_class == "minimum_footprint_passing_reduced_proxy"
    problem_level_native_claim = selection_class == "minimum_footprint_passing_native"
    record = {
        "slug": key["slug"],
        "category": row["category"],
        "selection_class": selection_class,
        "selection_role": selection_role,
        "source_matrix_row_key": {
            "slug": key["slug"],
            "source_workload_uuid": key["source_workload_uuid"],
            "variant_id": key["variant_id"],
        },
        "source_matrix_row_sha256": source_contract_sha256(row),
        "source_matrix_variant_sha256": row["variant_contract_sha256"],
        "selection_evidence": {
            "kind": evidence_kind,
            "sha256": evidence_sha256,
            "observed_latency_ms": latency_ms,
        },
        "definition_footprint": footprint,
        "selection_rank": {
            "aggregate_declared_tensor_bits": footprint[
                "aggregate_declared_tensor_bits"
            ],
            "peak_declared_tensor_bits": footprint[
                "peak_declared_tensor_bits"
            ],
            "aggregate_declared_tensor_elements": footprint[
                "aggregate_declared_tensor_elements"
            ],
            "resolved_axis_volume": footprint["resolved_axis_volume"],
            "scalar_control_tuple": footprint["scalar_control_tuple"],
            "source_workload_index": row["source_workload_index"],
            "source_workload_uuid": key["source_workload_uuid"],
            "variant_id": key["variant_id"],
        },
        "source_workload_index": row["source_workload_index"],
        "source_workload_uuid": key["source_workload_uuid"],
        "source_axes": row["source_axes"],
        "source_row_sha256": row["source_row_sha256"],
        "capture_workload_index": row["capture_workload_index"],
        "capture_axes": row["capture_axes"],
        "capture_row_sha256": row["capture_row_sha256"],
        "cohort": row["cohort"],
        "claim_scope": row["claim_scope"],
        "scale_modified": scale_modified,
        "native_shape": row["native_shape"],
        "native_predictive": row["native_predictive"],
        "native_scale_claim_eligible": not scale_modified,
        "problem_level_native_claim_eligible": problem_level_native_claim,
        "trace_contract": dict(REPRESENTATIVE_CAPTURE_CONTRACT),
    }
    _require(
        variant.get("variant_contract_sha256")
        == record["source_matrix_variant_sha256"],
        f"row/variant contract mismatch: {key!r}",
    )
    if scale_modified:
        provenance = variant.get("reduced_provenance")
        definition_modified = (
            isinstance(provenance, dict)
            and isinstance(provenance.get("native_definition_file_sha256"), str)
            and isinstance(provenance.get("reduced_definition_file_sha256"), str)
            and provenance["native_definition_file_sha256"]
            != provenance["reduced_definition_file_sha256"]
        )
        _require(
            record["native_predictive"] is False
            and record["native_scale_claim_eligible"] is False
            and (
                record["source_row_sha256"] != record["capture_row_sha256"]
                or definition_modified
            ),
            f"modified-scale row is not claim-safe: {key!r}",
        )
    return record


def validate_representative_capture_outcome(outcome: dict[str, Any]) -> None:
    """Validate the representative campaign's two-layer admission boundary.

    Static control evidence remains mandatory as an explicit outcome, but a
    closed ``origin_unresolved`` result does not invalidate independently
    complete NVBit and NCU evidence.  This deliberately does not modify the
    exhaustive campaign's older all-three-layer contract.
    """
    _require(isinstance(outcome, dict), "representative capture outcome is malformed")
    _require(
        set(outcome) == {"nvbit_trace", "ncu_profile", "control"},
        "representative capture outcome fields are not closed",
    )
    nvbit = outcome.get("nvbit_trace")
    ncu = outcome.get("ncu_profile")
    control = outcome.get("control")
    _require(
        isinstance(nvbit, dict) and set(nvbit) == {"status"}
        and nvbit.get("status") == "complete",
        "full unsampled NVBit trace is not complete",
    )
    _require(
        isinstance(ncu, dict) and set(ncu) == {"status", "metric_scope"}
        and ncu.get("status") == "complete"
        and ncu.get("metric_scope") == "cycles_only_binding",
        "cycles-only NCU profile is not complete",
    )
    _require(isinstance(control, dict), "control outcome is absent")
    status = control.get("status")
    if status == "complete":
        _require(
            set(control) == {"status", "reason"} and control.get("reason") is None,
            "complete control outcome is malformed",
        )
        return
    _require(
        status == "typed_unavailable_no_stable_static_module"
        and set(control) == {"status", "reason"}
        and control.get("reason") == "origin_unresolved",
        "control outcome is neither complete nor the closed typed-unavailable result",
    )


def _selection_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    rank = record["selection_rank"]
    return (
        rank["aggregate_declared_tensor_bits"],
        rank["peak_declared_tensor_bits"],
        rank["aggregate_declared_tensor_elements"],
        rank["resolved_axis_volume"],
        tuple(tuple(item) for item in rank["scalar_control_tuple"]),
        rank["source_workload_index"],
        rank["source_workload_uuid"],
        rank["variant_id"],
    )


def select_representatives(
    *, matrix: dict[str, Any], inventory: dict[str, Any],
    runnability: dict[str, Any], shrink: dict[str, Any],
    footprint_index: dict[tuple[str, str, str], dict[str, Any]],
    enforce_current_counts: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selected rows and typed architecture exclusions.

    The function is intentionally pure so fixture tests can exercise every
    tie-break and claim boundary without writing a campaign artifact.
    """
    variants = _index_variants(matrix)
    run_cells = _index_cells(runnability, label="runnability")
    shrink_cells = _index_cells(shrink, label="shrink")
    problems = inventory.get("problems")
    rows = matrix.get("rows")
    _require(isinstance(problems, list), "coverage inventory problems are absent")
    _require(isinstance(rows, list), "source matrix rows are absent")

    matrix_by_slug: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_row_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        _require(isinstance(row, dict) and isinstance(row.get("key"), dict),
                 "source matrix row is malformed")
        key = row["key"]
        identity = (key.get("slug"), key.get("source_workload_uuid"),
                    key.get("variant_id"))
        _require(
            all(isinstance(item, str) and item for item in identity)
            and identity not in seen_row_keys,
            f"source matrix row identity is duplicated/invalid: {identity!r}",
        )
        seen_row_keys.add(identity)
        matrix_by_slug[identity[0]].append(row)

    applicable: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    for problem in problems:
        _require(isinstance(problem, dict), "coverage problem is malformed")
        slug = problem.get("slug")
        _require(
            isinstance(slug, str) and slug and slug not in seen_problems,
            f"coverage problem slug is duplicated/invalid: {slug!r}",
        )
        seen_problems.add(slug)
        reason = problem.get("architecture_reason")
        if reason is None:
            applicable.append(problem)
            _require(slug in matrix_by_slug, f"applicable problem has no matrix row: {slug}")
        else:
            _require(reason in ARCHITECTURE_REASONS,
                     f"architecture reason is untyped: {slug}:{reason!r}")
            _require(slug not in matrix_by_slug,
                     f"architecture-inapplicable problem is scheduled: {slug}")
            excluded.append({
                "slug": slug,
                "category": problem.get("category"),
                "reason": reason,
                "source_workload_rows": problem.get("source_workload_row_count"),
                "inventory_problem_sha256": canonical_sha256(problem),
            })

    if enforce_current_counts:
        _require(len(problems) == EXPECTED_PROBLEMS,
                 f"problem count changed: {len(problems)}")
        _require(len(applicable) == EXPECTED_APPLICABLE_PROBLEMS,
                 f"applicable problem count changed: {len(applicable)}")
        _require(len(excluded) == EXPECTED_ARCH_INAPPLICABLE_PROBLEMS,
                 f"architecture exclusion count changed: {len(excluded)}")
        _require(
            sum(item["source_workload_rows"] for item in excluded)
            == EXPECTED_ARCH_INAPPLICABLE_ROWS,
            "architecture exclusion row count changed",
        )

    selections: list[dict[str, Any]] = []
    for problem in applicable:
        slug = problem["slug"]
        problem_rows = matrix_by_slug[slug]
        reduced = [row for row in problem_rows if row.get("cohort") == "reduced_proxy"]
        if reduced:
            _require(slug in shrink_cells,
                     f"reduced-proxy problem lacks shrink authority: {slug}")
            candidates = [
                _reduced_candidate(
                    row=row,
                    variant=variants[(slug, row["key"]["variant_id"])],
                    shrink_cell=shrink_cells[slug],
                    footprint=footprint_index[(
                        slug, row["key"]["source_workload_uuid"],
                        row["key"]["variant_id"],
                    )],
                )
                for row in reduced
            ]
            selections.append(min(candidates, key=_selection_sort_key))
            partial = [
                row for row in problem_rows
                if row.get("cohort") == "partial_native_row"
            ]
            if partial:
                _require(slug in run_cells,
                         f"partial-native problem lacks runnability authority: {slug}")
                companions = [
                    _partial_native_candidate(
                        row=row,
                        variant=variants[(slug, row["key"]["variant_id"])],
                        run_cell=run_cells[slug],
                        footprint=footprint_index[(
                            slug, row["key"]["source_workload_uuid"],
                            row["key"]["variant_id"],
                        )],
                    )
                    for row in partial
                ]
                selections.append(min(companions, key=_selection_sort_key))
        else:
            native = [row for row in problem_rows if row.get("cohort") == "native"]
            _require(native, f"applicable problem has neither reduced nor native rows: {slug}")
            _require(slug in run_cells, f"native problem lacks runnability authority: {slug}")
            candidates = [
                _native_candidate(
                    row=row,
                    variant=variants[(slug, row["key"]["variant_id"])],
                    run_cell=run_cells[slug],
                    footprint=footprint_index[(
                        slug, row["key"]["source_workload_uuid"],
                        row["key"]["variant_id"],
                    )],
                )
                for row in native
            ]
            selections.append(min(candidates, key=_selection_sort_key))

    role_order = {"application_coverage_anchor": 0, "native_comparison_companion": 1}
    selections.sort(key=lambda item: (
        item["category"], item["slug"], role_order[item["selection_role"]]
    ))
    excluded.sort(key=lambda item: (item["category"], item["slug"]))
    anchors = [
        item for item in selections
        if item["selection_role"] == "application_coverage_anchor"
    ]
    _require(
        len(anchors) == len(applicable)
        and len({item["slug"] for item in anchors}) == len(anchors)
        and {item["slug"] for item in anchors}
        == {item["slug"] for item in applicable},
        "application anchors are not one-to-one with applicable problems",
    )
    selection_keys = [
        (
            item["slug"], item["source_workload_uuid"],
            item["source_matrix_row_key"]["variant_id"],
        )
        for item in selections
    ]
    _require(len(selection_keys) == len(set(selection_keys)),
             "representative row selection is duplicated")
    if enforce_current_counts:
        cohort_counts = Counter(item["cohort"] for item in selections)
        _require(
            cohort_counts == {
                "native": EXPECTED_NATIVE_SELECTIONS,
                "reduced_proxy": EXPECTED_REDUCED_SELECTIONS,
                "partial_native_row": EXPECTED_PARTIAL_NATIVE_COMPANIONS,
            },
            f"representative cohort counts changed: {dict(cohort_counts)!r}",
        )
    return selections, excluded


def _chunks(selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cohort in ("native", "reduced_proxy", "partial_native_row"):
        cohort_rows = [row for row in selections if row["cohort"] == cohort]
        cohort_label = cohort.replace("_", "-")
        for start in range(0, len(cohort_rows), ROWS_PER_CHUNK):
            group = cohort_rows[start:start + ROWS_PER_CHUNK]
            result.append({
                "chunk_id": f"sol-representative-{cohort_label}-{start // ROWS_PER_CHUNK:03d}",
                "cohort": cohort,
                "selection_ids": [
                    {
                        "slug": row["slug"],
                        "source_workload_uuid": row["source_workload_uuid"],
                        "variant_id": row["source_matrix_row_key"]["variant_id"],
                    }
                    for row in group
                ],
            })
    return result


def build_matrix(
    *, source_matrix_path: Path, inventory_path: Path,
    runnability_path: Path, shrink_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    matrix, matrix_payload = _read_object(Path(source_matrix_path), label="source matrix")
    inventory, inventory_payload = _read_object(Path(inventory_path), label="coverage inventory")
    runnability, runnability_payload = _read_object(Path(runnability_path), label="runnability")
    shrink, shrink_payload = _read_object(Path(shrink_path), label="shrink authority")

    _schema_validate(matrix, repo_root=root, schema_path=FULL_MATRIX_SCHEMA_PATH,
                     label="source matrix")
    _schema_validate(inventory, repo_root=root, schema_path=INVENTORY_SCHEMA_PATH,
                     label="coverage inventory")
    _schema_validate(runnability, repo_root=root, schema_path=RUNNABILITY_SCHEMA_PATH,
                     label="runnability")
    _schema_validate(shrink, repo_root=root, schema_path=SHRINK_SCHEMA_PATH,
                     label="shrink authority")
    _verify_source_matrix(matrix)

    matrix_sha = sha256_bytes(matrix_payload)
    runnability_sha = sha256_bytes(runnability_payload)
    shrink_sha = sha256_bytes(shrink_payload)
    _require(
        inventory.get("inputs", {}).get("matrix", {}).get("sha256") == matrix_sha
        and inventory.get("inputs", {}).get("runnability", {}).get("sha256")
        == runnability_sha
        and inventory.get("inputs", {}).get("shrink", {}).get("sha256") == shrink_sha,
        "coverage inventory input hashes do not bind supplied authorities",
    )
    _require(
        matrix.get("runnability_authority", {}).get("sha256") == runnability_sha
        and matrix.get("reduced_proxy_authority", {}).get("sha256") == shrink_sha,
        "source matrix evidence pins do not bind supplied authorities",
    )
    source_commit = matrix.get("source_commit")
    _require(
        isinstance(source_commit, str) and len(source_commit) == 40
        and inventory.get("source_commit") == source_commit
        and runnability.get("source_commit") == source_commit
        and shrink.get("source_commit") == source_commit,
        "representative inputs have different source commits",
    )

    footprint_index = build_footprint_index(
        matrix=matrix,
        repo_root=root,
        shrink_root=Path(shrink_path).resolve().parent,
    )

    selections, exclusions = select_representatives(
        matrix=matrix,
        inventory=inventory,
        runnability=runnability,
        shrink=shrink,
        footprint_index=footprint_index,
        enforce_current_counts=True,
    )
    chunks = _chunks(selections)
    selection_identity = [
        {
            "selection_role": item["selection_role"],
            "source_matrix_row_key": item["source_matrix_row_key"],
        }
        for item in selections
    ]
    exclusion_identity = [
        {key: item[key] for key in ("slug", "category", "reason")}
        for item in exclusions
    ]
    category_counts = {
        category: {
            "selected": sum(item["category"] == category for item in selections),
            "application_coverage_anchors": sum(
                item["category"] == category
                and item["selection_role"] == "application_coverage_anchor"
                for item in selections
            ),
            "native_comparison_companions": sum(
                item["category"] == category
                and item["selection_role"] == "native_comparison_companion"
                for item in selections
            ),
            "architecture_inapplicable": sum(
                item["category"] == category for item in exclusions
            ),
        }
        for category in ("FIB", "L1", "L2", "Quant")
    }
    document: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "sm86_sol_representative_capture_matrix_v1",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "source_commit": source_commit,
        "input_pins": {
            "source_matrix": _input_pin(
                name=Path(source_matrix_path).name,
                payload=matrix_payload,
                owner=inventory["inputs"]["matrix"]["owner"],
            ),
            "coverage_inventory": _input_pin(
                name=Path(inventory_path).name,
                payload=inventory_payload,
            ),
            "runnability": _input_pin(
                name=Path(runnability_path).name,
                payload=runnability_payload,
                owner=inventory["inputs"]["runnability"]["owner"],
            ),
            "shrink": _input_pin(
                name=Path(shrink_path).name,
                payload=shrink_payload,
                owner=inventory["inputs"]["shrink"]["owner"],
            ),
        },
        "source_matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "selection_receipt": {
            "selector_path": Path(__file__).resolve().relative_to(root).as_posix(),
            "selector_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "schema_path": SCHEMA_PATH.as_posix(),
            "schema_sha256": sha256_bytes((root / SCHEMA_PATH).read_bytes()),
            "authority_matrix_sha256": matrix_sha,
            "authority_matrix_seal_sha256": matrix["matrix_seal_sha256"],
            "selection_identity_sha256": canonical_sha256(selection_identity),
            "architecture_exclusion_identity_sha256": canonical_sha256(
                exclusion_identity
            ),
        },
        "execution_boundary": {
            "authority_source_commit": source_commit,
            "authority_matrix_sha256": matrix_sha,
            "execution_source_commit": source_commit,
            "clean_head_required": True,
            "carry_forward_proof_required_if_execution_commit_differs": True,
            "filtered_execution_matrix_status": (
                "must_be_rebuilt_and_resealed_at_execution_commit"
            ),
        },
        "selection_policy": {
            "cardinality": (
                "exactly_one_application_anchor_per_sm86_applicable_problem_"
                "plus_one_partial_native_companion_when_available"
            ),
            "proxy_precedence": (
                "eligible_reduced_proxy_is_application_anchor_without_"
                "discarding_partial_native_companion"
            ),
            "candidate_order": (
                "aggregate_bits_then_peak_bits_then_aggregate_elements_then_"
                "axis_volume_then_scalar_tuple_then_source_index_then_uuid_"
                "then_variant_id"
            ),
            "passing_evidence_required": True,
            "observed_latency_used_for_ranking": False,
        },
        "capture_contract": dict(REPRESENTATIVE_CAPTURE_CONTRACT),
        "claim_policy": {
            "architecture_inapplicable_is_not_native_failure": True,
            "modified_scale_native_predictive": False,
            "modified_scale_native_scale_claims_forbidden": True,
            "representative_matrix_is_not_full_row_coverage": True,
            "trace_completeness_requires_done_evidence": True,
        },
        "counts": {
            "problems": len(inventory["problems"]),
            "applicable_problems": len({item["slug"] for item in selections}),
            "unique_applications_covered": len({
                item["slug"] for item in selections
            }),
            "application_coverage_anchors": sum(
                item["selection_role"] == "application_coverage_anchor"
                for item in selections
            ),
            "native_comparison_companions": sum(
                item["selection_role"] == "native_comparison_companion"
                for item in selections
            ),
            "architecture_inapplicable_problems": len(exclusions),
            "architecture_inapplicable_rows": sum(
                item["source_workload_rows"] for item in exclusions
            ),
            "selected_rows": len(selections),
            "native_selected_rows": sum(
                item["cohort"] == "native" for item in selections
            ),
            "partial_native_selected_rows": sum(
                item["cohort"] == "partial_native_row" for item in selections
            ),
            "reduced_proxy_selected_rows": sum(
                item["cohort"] == "reduced_proxy" for item in selections
            ),
            "modified_scale_rows": sum(item["scale_modified"] for item in selections),
            "native_predictive_false_rows": sum(
                item["native_predictive"] is False for item in selections
            ),
            "chunks": len(chunks),
        },
        "by_category": category_counts,
        "architecture_exclusions": exclusions,
        "selections": selections,
        "chunks": chunks,
    }
    document["matrix_seal_sha256"] = canonical_sha256(document)
    _schema_validate(document, repo_root=root, schema_path=SCHEMA_PATH,
                     label="representative matrix")
    return document


def _deliver(document: dict[str, Any], *, emit: Path | None, check: Path | None) -> int:
    _require((emit is None) != (check is None),
             "exactly one of --emit/--check is required")
    expected = canonical_bytes(document)
    path = emit if emit is not None else check
    assert path is not None
    if check is not None:
        try:
            observed = path.read_bytes()
        except OSError as exc:
            raise RepresentativeMatrixError(f"cannot read checked artifact: {exc}") from exc
        _require(observed == expected, "representative matrix bytes differ from regeneration")
        print(
            "SOL representative matrix verified: "
            f"rows={document['counts']['selected_rows']} "
            f"excluded={document['counts']['architecture_inapplicable_problems']}"
        )
        return 0
    assert emit is not None
    emit.parent.mkdir(parents=True, exist_ok=True)
    emit.write_bytes(expected)
    print(
        "SOL representative matrix emitted: "
        f"rows={document['counts']['selected_rows']} -> {emit}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-matrix", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--runnability", type=Path, required=True)
    parser.add_argument("--shrink", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", type=Path)
    mode.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    try:
        document = build_matrix(
            source_matrix_path=args.source_matrix,
            inventory_path=args.inventory,
            runnability_path=args.runnability,
            shrink_path=args.shrink,
        )
        return _deliver(document, emit=args.emit, check=args.check)
    except RepresentativeMatrixError as exc:
        print(f"build_sol_representative_capture_matrix: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
