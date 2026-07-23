#!/usr/bin/env python3
"""Build strict, non-native SOL proxies for unsampled NVBit tracing.

This module deliberately does *not* infer trace infeasibility from ordinary
SOL runnability, native OOM, a timeout, or workload size.  A caller must supply
one closed, hash-bound target record whose measurement says that the smallest
native workload is infeasible with sampling disabled.  The target also carries
the reviewed axis update.  Only definition-declared variable axes may change;
constant/expression axes can change solely through an existing reviewed
``sol_shrink_apply.MODEL_SPECS`` definition policy.

The native definition and workload are read-only inputs.  A candidate is a
separate definition plus a one-row workload, both byte-bound to their sources.
It becomes publishable only after the existing SOL certified runner reports a
complete runnable/correctness-passing result over those exact bytes.  Published
records always state ``native_predictive=false`` and
``claim_scope=trace_scale_proxy_only``.

No GPU work occurs while importing or using the pure planning/validation
helpers.  ``certify_with_sol`` is the explicit execution boundary and is kept
separate so callers and tests cannot trigger a live run accidentally.
"""
from __future__ import annotations

import copy
import ast
import hashlib
import json
import os
import operator
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]

from tools.local_labels import sol_reconstitution as sr  # noqa: E402
from tools.local_labels import sol_shrink_apply as shrink  # noqa: E402


ARTIFACT_ID = "sm86_sol_trace_scale_proxy_external_v1"
SCHEMA_VERSION = 1
CLAIM_SCOPE = "trace_scale_proxy_only"
ROW_SHA256_SCOPE = "exact_json_line_bytes_excluding_lf_only"
SCHEMA_PATH = (
    REPO_ROOT
    / "workloads/simbench_bw/schemas/sol_trace_scale_proxy.external.v1.schema.json"
)

# These tokens denote an explicit unsampled-trace capacity result.  A generic
# subprocess failure (including SIGTERM) is intentionally absent: it does not
# prove that the workload itself is infeasible.
TRACE_INFEASIBLE_REASONS = frozenset({
    "trace_target_failed_timeout",
    "trace_target_failed_tree_limit_exceeded",
    "trace_volume_exceeded",
    "nvbit_slowdown_budget_exceeded",
})

TARGET_KEYS = frozenset({
    "axis_updates",
    "category",
    "evidence_sha256",
    "language",
    "native_definition_sha256",
    "native_row_sha256",
    "partial_trace_allowed",
    "reviewed_definition_policy",
    "sampling_policy",
    "slug",
    "smallest_native_workload",
    "source_workload_index",
    "source_workload_uuid",
    "trace_failure_reason",
})

VALIDATION_KEYS = frozenset({
    "correctness_result",
    "definition_sha256",
    "evidence_sha256",
    "execution_evidence",
    "method",
    "solution_sha256",
    "status",
    "workload_sha256",
})

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

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class TraceScaleProxyError(ValueError):
    """A trace proxy contract or source binding failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise TraceScaleProxyError(detail)


def _source_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_source_rows(path: Path) -> list[tuple[dict[str, Any], bytes, int]]:
    """Delegate the byte-sensitive JSONL grammar to the shrink authority."""
    try:
        return shrink._load_source_rows(path)
    except (OSError, ValueError) as exc:
        raise TraceScaleProxyError(str(exc)) from exc


def _resolve_expression(expression: str, values: Mapping[str, int]) -> int:
    """Mirror SOL-ExecBench's closed arithmetic shape-expression grammar."""
    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(
                node.value, bool
            ):
                return node.value
            raise TraceScaleProxyError("shape expression has a nonnumeric constant")
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise KeyError(node.id)
            value = values[node.id]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TraceScaleProxyError("shape expression variable is not an integer")
            return value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](evaluate(node.operand))
        raise TraceScaleProxyError(
            f"unsupported shape expression node: {type(node).__name__}"
        )

    try:
        root = ast.parse(expression, mode="eval")
        result = evaluate(root.body)
    except (SyntaxError, ArithmeticError) as exc:
        raise TraceScaleProxyError(f"invalid shape expression: {exc}") from exc
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    _require(
        isinstance(result, int) and not isinstance(result, bool),
        "shape expression did not resolve to an integer",
    )
    return result


def _resolved_axes(
    definition: Mapping[str, Any], workload_axes: Mapping[str, Any],
) -> tuple[dict[str, int], list[list[Any]]]:
    contracts = definition.get("axes")
    inputs = definition.get("inputs")
    _require(isinstance(contracts, dict), "definition axes is not an object")
    _require(isinstance(inputs, dict), "definition inputs is not an object")
    values: dict[str, int] = {}
    var_names: set[str] = set()
    pending: dict[str, str] = {}
    for name, contract in contracts.items():
        _require(isinstance(name, str) and isinstance(contract, dict), "axis contract is invalid")
        kind = contract.get("type")
        if kind == "const":
            value = contract.get("value")
            _require(
                isinstance(value, int) and not isinstance(value, bool),
                f"constant axis is not an integer: {name}",
            )
            values[name] = value
        elif kind == "var":
            var_names.add(name)
        elif kind == "expr":
            expression = contract.get("expression")
            _require(isinstance(expression, str) and bool(expression), f"expression axis is invalid: {name}")
            pending[name] = expression
        else:
            raise TraceScaleProxyError(f"unknown definition axis kind: {name}")
    _require(var_names <= set(workload_axes), "workload omits a definition variable axis")
    for name in var_names:
        value = workload_axes[name]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"workload variable axis is not a nonnegative integer: {name}",
        )
        values[name] = value
    extra: list[list[Any]] = []
    for name in sorted(set(workload_axes) - var_names):
        spec = inputs.get(name)
        value = workload_axes[name]
        _require(
            isinstance(spec, dict)
            and spec.get("shape") is None
            and isinstance(value, int)
            and not isinstance(value, bool),
            f"extra workload axis is not a declared scalar input: {name}",
        )
        extra.append([name, value])
    while pending:
        progress = False
        for name in list(pending):
            try:
                value = _resolve_expression(pending[name], values)
            except KeyError:
                continue
            values[name] = value
            del pending[name]
            progress = True
        _require(progress, "definition expression axes have an unresolved dependency")
    _require(
        all(isinstance(value, int) and value >= 0 for value in values.values()),
        "resolved definition axis is negative",
    )
    return values, extra


def execution_footprint(
    definition: Mapping[str, Any], workload_axes: Mapping[str, Any], *,
    source_index: int, source_uuid: str, variant: str = "native",
) -> dict[str, Any]:
    """Return the deterministic, definition-aware execution-footprint key.

    This is a selector, not an NVBit-volume predictor.  It prevents latency and
    source order from masquerading as workload size while retaining them only
    as deterministic final tie breakers.
    """
    resolved, extras = _resolved_axes(definition, workload_axes)
    tensor_bits: list[int] = []
    tensor_elements: list[int] = []
    for side in ("inputs", "outputs"):
        specs = definition.get(side)
        _require(isinstance(specs, dict), f"definition {side} is not an object")
        for name, spec in specs.items():
            _require(isinstance(spec, dict), f"definition tensor is invalid: {side}/{name}")
            bits = DTYPE_BITS.get(spec.get("dtype"))
            _require(bits is not None, f"definition tensor dtype is unsupported: {side}/{name}")
            shape = spec.get("shape")
            if shape is None:
                elements = 1
            else:
                _require(isinstance(shape, list), f"definition tensor shape is invalid: {side}/{name}")
                elements = 1
                for component in shape:
                    _require(isinstance(component, str) and bool(component), "tensor shape component is invalid")
                    if component.isdigit():
                        dimension = int(component)
                    else:
                        _require(component in resolved, f"tensor shape references unknown axis: {component}")
                        dimension = resolved[component]
                    _require(dimension >= 0, "tensor shape dimension is negative")
                    elements *= dimension
            tensor_elements.append(elements)
            tensor_bits.append(elements * bits)
    axes_product = 1
    for value in resolved.values():
        axes_product *= max(1, value)
    components = {
        "aggregate_tensor_bits": sum(tensor_bits),
        "peak_tensor_bits": max(tensor_bits, default=0),
        "aggregate_elements": sum(tensor_elements),
        "resolved_axes_product": axes_product,
        "extra_scalar_axes": extras,
        "source_workload_index": source_index,
        "source_workload_uuid": source_uuid,
        "variant": variant,
    }
    components["key_sha256"] = sha256_bytes(canonical_json_bytes(components))
    return components


def _footprint_sort_key(footprint: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        footprint["aggregate_tensor_bits"],
        footprint["peak_tensor_bits"],
        footprint["aggregate_elements"],
        footprint["resolved_axes_product"],
        tuple(tuple(value) for value in footprint["extra_scalar_axes"]),
        footprint["source_workload_index"],
        footprint["source_workload_uuid"],
        footprint["variant"],
    )


def validate_target(target: Mapping[str, Any]) -> None:
    """Validate one explicit, measured trace-infeasibility target.

    ``axis_updates`` is a reviewed decision, not a measurement.  It is checked
    against the live definition by :func:`plan_candidate` before use.
    """
    _require(set(target) == TARGET_KEYS, "trace target has an open/unknown field set")
    for field in ("category", "slug", "language", "source_workload_uuid"):
        _require(
            isinstance(target.get(field), str) and bool(target[field]),
            f"trace target {field} is invalid",
        )
    _require(
        isinstance(target.get("source_workload_index"), int)
        and not isinstance(target["source_workload_index"], bool)
        and target["source_workload_index"] >= 0,
        "trace target source_workload_index is invalid",
    )
    for field in (
        "native_definition_sha256",
        "native_row_sha256",
        "evidence_sha256",
    ):
        _require(_is_sha256(target.get(field)), f"trace target {field} is invalid")
    _require(
        target.get("smallest_native_workload") is True,
        "trace target is not the measured smallest native workload",
    )
    _require(
        target.get("sampling_policy") == "off",
        "trace target was not measured with sampling disabled",
    )
    _require(
        target.get("partial_trace_allowed") is False,
        "trace target allowed a partial trace",
    )
    _require(
        target.get("trace_failure_reason") in TRACE_INFEASIBLE_REASONS,
        "trace target lacks a typed unsampled-capacity failure",
    )
    updates = target.get("axis_updates")
    _require(isinstance(updates, dict) and bool(updates), "axis_updates is empty")
    _require(
        all(isinstance(name, str) and name for name in updates),
        "axis_updates contains an invalid axis name",
    )
    policy = target.get("reviewed_definition_policy")
    _require(
        policy is None or (isinstance(policy, str) and bool(policy)),
        "reviewed_definition_policy is invalid",
    )


def _definition_variant(
    *, definition: dict[str, Any], slug: str, policy: str | None,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Return a separate effective definition and its closed delta.

    Even when constants do not change, the candidate gets a separate definition
    child.  The bytes are then exactly the native definition bytes, rather than
    a parse/reserialize that could reorder SOL's positional input contract.
    """
    if policy is None:
        raise AssertionError("native definition bytes are selected by the caller")
    spec = shrink.MODEL_SPECS.get(slug)
    _require(spec is not None, "reviewed definition policy has no registered slug")
    _require(
        spec.get("policy") == policy,
        "reviewed definition policy does not match the registered slug policy",
    )
    try:
        payload, delta = shrink._definition_variant(definition, spec)
    except ValueError as exc:
        raise TraceScaleProxyError(str(exc)) from exc
    _require(payload is not None and bool(delta), "reviewed policy changed no definition")
    return payload, delta


def _validate_axis_updates(
    definition: Mapping[str, Any], native_axes: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    contracts = definition.get("axes")
    _require(isinstance(contracts, dict), "definition axes is not an object")
    _require(
        set(native_axes) <= set(contracts),
        "native workload contains an axis absent from the definition",
    )
    reduced = copy.deepcopy(dict(native_axes))
    for name, value in updates.items():
        contract = contracts.get(name)
        _require(
            isinstance(contract, dict) and contract.get("type") == "var",
            f"trace proxy axis is not a definition-declared var: {name}",
        )
        native = native_axes.get(name)
        _require(
            isinstance(native, int) and not isinstance(native, bool) and native > 0,
            f"trace proxy native axis is not a positive integer: {name}",
        )
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"trace proxy reduced axis is not a positive integer: {name}",
        )
        _require(value < native, f"trace proxy axis does not reduce its source: {name}")
        reduced[name] = value
    _require(set(reduced) == set(native_axes), "trace proxy changed the axis key set")
    return reduced


def _axes_delta(
    native: Mapping[str, Any], reduced: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"axis": name, "from": native[name], "to": reduced[name]}
        for name in sorted(native)
        if native[name] != reduced[name]
    ]


def plan_candidate(
    *, cell: Mapping[str, Any], benchmark_dir: Path, solution_path: Path,
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Plan one byte-bound trace proxy without running or writing anything."""
    validate_target(target)
    for field in ("category", "slug", "language"):
        _require(cell.get(field) == target[field], f"cell/target {field} mismatch")

    definition_path = Path(benchmark_dir) / "definition.json"
    workload_path = Path(benchmark_dir) / "workload.jsonl"
    try:
        definition_bytes = definition_path.read_bytes()
        workload_file_bytes = workload_path.read_bytes()
        solution_bytes = Path(solution_path).read_bytes()
        definition = json.loads(definition_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceScaleProxyError(f"trace proxy source is unreadable/invalid: {exc}") from exc
    _require(isinstance(definition, dict), "definition root is not an object")
    _require(
        sha256_bytes(definition_bytes) == target["native_definition_sha256"],
        "native definition bytes differ from the measured target",
    )
    rows = _read_source_rows(workload_path)
    index = target["source_workload_index"]
    _require(index < len(rows), "source_workload_index is outside the source JSONL")
    source_row, source_bytes, source_line = rows[index]
    _require(
        source_row.get("uuid") == target["source_workload_uuid"],
        "source workload UUID differs from the measured target",
    )
    _require(
        sha256_bytes(source_bytes) == target["native_row_sha256"],
        "source workload bytes differ from the measured target",
    )
    native_axes = source_row.get("axes")
    _require(isinstance(native_axes, dict), "source workload lacks an axes object")
    native_footprints = []
    for row_index, (row, _raw, _line) in enumerate(rows):
        row_axes = row.get("axes")
        row_uuid = row.get("uuid")
        _require(
            isinstance(row_axes, dict) and isinstance(row_uuid, str) and bool(row_uuid),
            "source workload row lacks footprint identity",
        )
        native_footprints.append(
            execution_footprint(
                definition,
                row_axes,
                source_index=row_index,
                source_uuid=row_uuid,
            )
        )
    smallest = min(native_footprints, key=_footprint_sort_key)
    _require(
        smallest["source_workload_index"] == index
        and smallest["source_workload_uuid"] == target["source_workload_uuid"],
        "trace target is not the definition-aware smallest native workload",
    )
    native_footprint = native_footprints[index]
    reduced_axes = _validate_axis_updates(
        definition, native_axes, target["axis_updates"]
    )
    reduced_row = copy.deepcopy(source_row)
    reduced_row["axes"] = reduced_axes
    _require(
        set(reduced_row) == set(source_row)
        and all(
            reduced_row[key] == source_row[key]
            for key in source_row
            if key != "axes"
        ),
        "trace proxy changed a non-axis workload contract",
    )
    reduced_row_bytes = shrink._row_bytes(reduced_row) + b"\n"

    policy = target["reviewed_definition_policy"]
    if policy is None:
        reduced_definition_bytes = definition_bytes
        definition_delta: list[dict[str, Any]] = []
    else:
        reduced_definition_bytes, definition_delta = _definition_variant(
            definition=definition, slug=target["slug"], policy=policy
        )

    try:
        reduced_definition = json.loads(reduced_definition_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceScaleProxyError(f"reduced definition is invalid: {exc}") from exc
    _require(isinstance(reduced_definition, dict), "reduced definition root is invalid")
    reduced_footprint = execution_footprint(
        reduced_definition,
        reduced_axes,
        source_index=index,
        source_uuid=target["source_workload_uuid"],
        variant="trace_proxy",
    )
    _require(
        reduced_footprint["aggregate_tensor_bits"]
        < native_footprint["aggregate_tensor_bits"],
        "trace proxy does not reduce the definition tensor footprint",
    )

    axis_delta = _axes_delta(native_axes, reduced_axes)
    _require(bool(axis_delta) or bool(definition_delta), "trace proxy equals its source")
    identity = {
        "artifact_contract": ARTIFACT_ID,
        "definition_sha256": sha256_bytes(reduced_definition_bytes),
        "native_definition_sha256": sha256_bytes(definition_bytes),
        "native_row_sha256": sha256_bytes(source_bytes),
        "slug": target["slug"],
        "source_workload_index": index,
        "source_workload_uuid": target["source_workload_uuid"],
        "trace_evidence_sha256": target["evidence_sha256"],
        "workload_sha256": sha256_bytes(reduced_row_bytes),
    }
    proxy_id = sha256_bytes(canonical_json_bytes(identity))
    prefix = f"{target['category']}__{target['slug']}__trace_proxy__{proxy_id}"
    return {
        "proxy_id": proxy_id,
        "category": target["category"],
        "slug": target["slug"],
        "language": target["language"],
        "claim_scope": CLAIM_SCOPE,
        "native_predictive": False,
        "source_workload_index": index,
        "source_workload_line": source_line,
        "source_workload_uuid": target["source_workload_uuid"],
        "native_definition": {
            "path": _source_ref(definition_path),
            "sha256": sha256_bytes(definition_bytes),
            "size": len(definition_bytes),
        },
        "native_workload_file": {
            "path": _source_ref(workload_path),
            "sha256": sha256_bytes(workload_file_bytes),
            "size": len(workload_file_bytes),
        },
        "native_workload_row": {
            "sha256": sha256_bytes(source_bytes),
            "size": len(source_bytes),
            "sha256_scope": ROW_SHA256_SCOPE,
        },
        "solution": {
            "path": _source_ref(Path(solution_path)),
            "sha256": sha256_bytes(solution_bytes),
            "size": len(solution_bytes),
        },
        "trace_infeasibility": {
            "evidence_sha256": target["evidence_sha256"],
            "failure_reason": target["trace_failure_reason"],
            "partial_trace_allowed": False,
            "sampling_policy": "off",
            "smallest_native_workload": True,
        },
        "source_selection": {
            "method": "definition_tensor_footprint_v1",
            "rank": 0,
            "native_candidate_count": len(rows),
            "footprint": native_footprint,
        },
        "native_axes": copy.deepcopy(native_axes),
        "reduced_axes": reduced_axes,
        "reduced_footprint": reduced_footprint,
        "axes_delta": axis_delta,
        "definition_delta": definition_delta,
        "reduced_definition": {
            "name": f"{prefix}.definition.json",
            "sha256": sha256_bytes(reduced_definition_bytes),
            "size": len(reduced_definition_bytes),
        },
        "reduced_workload": {
            "name": f"{prefix}.workload.jsonl",
            "sha256": sha256_bytes(reduced_row_bytes),
            "size": len(reduced_row_bytes),
        },
        "_definition_bytes": reduced_definition_bytes,
        "_workload_bytes": reduced_row_bytes,
        "_cell": copy.deepcopy(dict(cell)),
    }


def plan_targets(
    targets: list[Mapping[str, Any]], *,
    cells: list[Mapping[str, Any]] | None = None,
    resolver: Callable[[Mapping[str, Any]], tuple[Path, Path, Path]] = sr.resolve_paths,
) -> list[dict[str, Any]]:
    """Plan every measured target exactly once against the sm_86 cell roster."""
    _require(bool(targets), "trace target set is empty")
    roster = list(cells) if cells is not None else sr.in_scope_cells()
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for cell in roster:
        key = (cell.get("category"), cell.get("slug"), cell.get("language"))
        _require(
            all(isinstance(value, str) and bool(value) for value in key),
            "sm86 cell roster contains an invalid identity",
        )
        _require(key not in by_key, "sm86 cell roster contains a duplicate identity")
        by_key[key] = cell
    planned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        validate_target(target)
        key = (target["category"], target["slug"], target["language"])
        _require(key not in seen, "trace target set contains a duplicate problem")
        seen.add(key)
        cell = by_key.get(key)
        _require(cell is not None, "trace target is outside the sm86 cell roster")
        try:
            _baseline, solution, benchmark = resolver(cell)
        except (OSError, FileNotFoundError, ValueError) as exc:
            raise TraceScaleProxyError(f"trace target source resolution failed: {exc}") from exc
        planned.append(
            plan_candidate(
                cell=cell,
                benchmark_dir=benchmark,
                solution_path=solution,
                target=target,
            )
        )
    planned.sort(key=lambda row: (row["category"], row["slug"], row["language"]))
    _require(len(planned) == len(targets), "trace target planning lost a target")
    return planned


def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in candidate.items()
        if not key.startswith("_")
    }


def validate_certification(
    candidate: Mapping[str, Any], certification: Mapping[str, Any],
) -> None:
    """Require a complete SOL full-correctness result over the exact proxy."""
    _require(
        set(certification) == VALIDATION_KEYS,
        "SOL proxy certification has an open/unknown field set",
    )
    _require(
        certification.get("method") == "sol_execbench_full_correctness",
        "SOL proxy certification method is not full correctness",
    )
    _require(certification.get("status") == sr.RUNNABLE, "SOL proxy is not runnable")
    _require(
        certification.get("correctness_result") == "passed",
        "SOL proxy correctness did not pass",
    )
    _require(
        certification.get("workload_sha256")
        == candidate["reduced_workload"]["sha256"],
        "SOL proxy certification used different workload bytes",
    )
    _require(
        certification.get("definition_sha256")
        == candidate["reduced_definition"]["sha256"],
        "SOL proxy certification used different definition bytes",
    )
    _require(
        certification.get("solution_sha256") == candidate["solution"]["sha256"],
        "SOL proxy certification used different solution bytes",
    )
    _require(
        _is_sha256(certification.get("evidence_sha256")),
        "SOL proxy certification evidence hash is invalid",
    )
    evidence = certification.get("execution_evidence")
    _require(isinstance(evidence, dict), "SOL proxy execution evidence is absent")
    _require(
        sha256_bytes(canonical_json_bytes(evidence))
        == certification["evidence_sha256"],
        "SOL proxy execution evidence hash mismatch",
    )


def certify_candidate(
    candidate: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Invoke an explicit validator and return the publishable public record."""
    certification = validator(candidate)
    _require(isinstance(certification, Mapping), "SOL proxy validator returned no record")
    validate_certification(candidate, certification)
    record = _public_candidate(candidate)
    record["validation"] = copy.deepcopy(dict(certification))
    return record


def build_authority(
    *, source_commit: str, target_manifest_sha256: str,
    candidates: list[Mapping[str, Any]],
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Certify candidates and build one closed authority plus child payloads."""
    _require(
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(char in "0123456789abcdef" for char in source_commit),
        "source_commit is not a full lowercase Git object ID",
    )
    _require(_is_sha256(target_manifest_sha256), "target manifest hash is invalid")
    _require(bool(candidates), "trace proxy candidate set is empty")
    ordered = sorted(candidates, key=lambda row: (row["category"], row["slug"]))
    _require(
        len({row["slug"] for row in ordered}) == len(ordered),
        "trace proxy candidate slugs are not unique",
    )
    children: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    for candidate in ordered:
        record = certify_candidate(candidate, validator)
        records.append(record)
        for field, private in (
            ("reduced_definition", "_definition_bytes"),
            ("reduced_workload", "_workload_bytes"),
        ):
            name = candidate[field]["name"]
            payload = candidate[private]
            _require(name not in children, "trace proxy child name collision")
            _require(
                sha256_bytes(payload) == candidate[field]["sha256"]
                and len(payload) == candidate[field]["size"],
                "trace proxy child bytes changed after planning",
            )
            children[name] = payload
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "arch": "sm_86",
        "gpu": "RTX 3090 Ti",
        "source_commit": source_commit,
        "target_manifest_sha256": target_manifest_sha256,
        "claim_scope": CLAIM_SCOPE,
        "native_predictive": False,
        "proxy_count": len(records),
        "proxies": records,
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)
    return artifact, children


def certify_with_sol(
    candidate: Mapping[str, Any], *, probe_contract: Mapping[str, Any],
    timeout: float = sr.DEFAULT_OUTER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Explicit GPU boundary: run the exact candidate via ``run_certified``.

    This function is never called by planning, schema validation, publication,
    or unit tests.  The caller opts into GPU work by invoking it directly.
    """
    with tempfile.TemporaryDirectory(prefix="sol-trace-proxy-cert-") as tmp:
        root = Path(tmp)
        definition = root / candidate["reduced_definition"]["name"]
        workload = root / candidate["reduced_workload"]["name"]
        definition.write_bytes(candidate["_definition_bytes"])
        workload.write_bytes(candidate["_workload_bytes"])
        result = sr.run_certified(
            candidate["_cell"],
            timeout=timeout,
            probe_contract=probe_contract,
            workload_path=workload,
            definition_path=definition,
        )
    evidence = copy.deepcopy(result)
    diagnostics = result.get("workload_diagnostics")
    passed = (
        result.get("status") == sr.RUNNABLE
        and result.get("diagnostics_complete") is True
        and isinstance(diagnostics, list)
        and len(diagnostics) == 1
        and diagnostics[0].get("diagnostic_present") is True
        and str(diagnostics[0].get("status", "")).lower() == "passed"
    )
    return {
        "method": "sol_execbench_full_correctness",
        "status": result.get("status"),
        "correctness_result": "passed" if passed else "failed",
        "workload_sha256": candidate["reduced_workload"]["sha256"],
        "definition_sha256": candidate["reduced_definition"]["sha256"],
        "solution_sha256": candidate["solution"]["sha256"],
        "evidence_sha256": sha256_bytes(canonical_json_bytes(evidence)),
        "execution_evidence": evidence,
    }


def publish_authority(
    *, output_root: Path, output_name: str, artifact: Mapping[str, Any],
    children: Mapping[str, bytes],
) -> Path:
    """Publish a complete authority into a fresh, bound external directory."""
    _require(
        isinstance(output_name, str)
        and output_name.endswith(".json")
        and "/" not in output_name
        and "\\" not in output_name,
        "trace proxy authority output name is unsafe",
    )
    try:
        identity = sr.validate_external_campaign_root(Path(output_root))
        descriptor = sr.open_external_root(identity)
        try:
            names = os.listdir(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, sr.PrerequisiteContractError) as exc:
        raise TraceScaleProxyError(str(exc)) from exc
    _require(not names, "trace proxy output root is not empty")
    for name in sorted(children):
        sr.publish_external_child(identity, name, children[name])
    payload = canonical_json_bytes(artifact)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(payload), schema)
    return sr.publish_external_child(identity, output_name, payload)
