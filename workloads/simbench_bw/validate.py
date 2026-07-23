#!/usr/bin/env python3
"""Artifact schema validator for SimBench-BW.

Walks committed artifacts (roster.toml, shapes/*.toml,
platform_matrix.toml, splits.toml, tolerances.toml, trace_budgets.toml,
suite_results.json, measurement_manifest.json, historical CUPTI Activity
sidecars,
NVBit phase ledgers, dual-run manifests, l4_checkpoints/*.json, and
cross-gen notes embedded in roster rows) and checks them against the
matching schema file under workloads/simbench_bw/schemas/.

Exit codes:
    0  all validated artifacts conform.
    1  at least one artifact failed validation.
    2  usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable


SUITE_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = SUITE_ROOT / "schemas"


# --- Minimal JSON-Schema-2020-12 validator ------------------------------------
#
# Implements the subset actually used by schemas under SUITE_ROOT/schemas.
# External jsonschema dependency is avoided to keep the harness installable
# on minimal cloud images.


class ValidationError(Exception):
    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


def _load_schema(schema_ref: Path) -> dict[str, Any]:
    key = str(schema_ref.resolve())
    if key not in _SCHEMA_CACHE:
        with open(schema_ref) as f:
            _SCHEMA_CACHE[key] = json.load(f)
    return _SCHEMA_CACHE[key]


def _resolve_ref(ref: str, root_schema: dict[str, Any], root_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Resolve $ref to (schema_node, root_schema_for_sub_refs, path_of_root)."""
    if ref.startswith("#/"):
        node = root_schema
        for part in ref[2:].split("/"):
            node = node[part]
        return node, root_schema, root_path
    if ref.endswith(".json"):
        sibling = (root_path.parent / ref).resolve()
        sub = _load_schema(sibling)
        return sub, sub, sibling
    raise ValueError(f"unsupported $ref form: {ref}")


def _type_matches(value: Any, t: str) -> bool:
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    if t == "null":
        return value is None
    raise ValueError(f"unsupported type: {t}")


def validate(
    instance: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any] | None = None,
    root_path: Path | None = None,
    path: str = "$",
) -> None:
    """Validate `instance` against `schema`. Raises ValidationError on the first failure."""
    if root_schema is None:
        root_schema = schema
    if root_path is None:
        raise ValueError("root_path is required")

    if "$ref" in schema:
        sub_schema, sub_root, sub_path = _resolve_ref(schema["$ref"], root_schema, root_path)
        validate(instance, sub_schema, root_schema=sub_root, root_path=sub_path, path=path)
        return

    if "const" in schema and instance != schema["const"]:
        raise ValidationError(path, f"expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise ValidationError(path, f"value {instance!r} not in enum {schema['enum']!r}")

    t = schema.get("type")
    if t is not None:
        if isinstance(t, list):
            if not any(_type_matches(instance, tt) for tt in t):
                raise ValidationError(path, f"type {type(instance).__name__!r} not in {t}")
        else:
            if not _type_matches(instance, t):
                raise ValidationError(path, f"expected type {t}, got {type(instance).__name__}")

    if isinstance(instance, str):
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            raise ValidationError(path, f"string {instance!r} does not match pattern {schema['pattern']!r}")
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise ValidationError(path, f"string too short ({len(instance)} < {schema['minLength']})")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise ValidationError(path, f"string too long ({len(instance)} > {schema['maxLength']})")
        if "format" in schema and schema["format"] == "date-time":
            if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", instance):
                raise ValidationError(path, f"not a date-time: {instance!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValidationError(path, f"value {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ValidationError(path, f"value {instance} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise ValidationError(path, f"value {instance} not > exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            raise ValidationError(path, f"value {instance} not < exclusiveMaximum {schema['exclusiveMaximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise ValidationError(path, f"array too short ({len(instance)} < {schema['minItems']})")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ValidationError(path, f"array too long ({len(instance)} > {schema['maxItems']})")
        if schema.get("uniqueItems"):
            seen = []
            for v in instance:
                if v in seen:
                    raise ValidationError(path, f"duplicate item {v!r}")
                seen.append(v)
        prefix_items = schema.get("prefixItems")
        items_schema = schema.get("items")
        if prefix_items is not None:
            for i, item_schema in enumerate(prefix_items):
                if i < len(instance):
                    validate(instance[i], item_schema, root_schema=root_schema, root_path=root_path, path=f"{path}[{i}]")
            if items_schema is False:
                if len(instance) > len(prefix_items):
                    raise ValidationError(path, f"array has trailing items after prefixItems")
            elif isinstance(items_schema, dict):
                for i in range(len(prefix_items), len(instance)):
                    validate(instance[i], items_schema, root_schema=root_schema, root_path=root_path, path=f"{path}[{i}]")
        elif items_schema is not None:
            if items_schema is False:
                if len(instance) > 0:
                    raise ValidationError(path, "array must be empty when items=false")
            else:
                for i, item in enumerate(instance):
                    validate(item, items_schema, root_schema=root_schema, root_path=root_path, path=f"{path}[{i}]")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                raise ValidationError(path, f"missing required key {key!r}")
        props = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        property_names_schema = schema.get("propertyNames")
        for key, value in instance.items():
            if property_names_schema is not None:
                validate(key, property_names_schema, root_schema=root_schema, root_path=root_path, path=f"{path}.{key}(key)")
            if key in props:
                validate(value, props[key], root_schema=root_schema, root_path=root_path, path=f"{path}.{key}")
            else:
                if additional is False:
                    raise ValidationError(path, f"unexpected key {key!r}")
                if isinstance(additional, dict):
                    validate(value, additional, root_schema=root_schema, root_path=root_path, path=f"{path}.{key}")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            raise ValidationError(path, f"object has too few properties ({len(instance)} < {schema['minProperties']})")

    for combinator in ("allOf", "anyOf", "oneOf"):
        if combinator not in schema:
            continue
        if combinator == "oneOf":
            matches = 0
            last_err: ValidationError | None = None
            for sub in schema["oneOf"]:
                try:
                    validate(instance, sub, root_schema=root_schema, root_path=root_path, path=path)
                    matches += 1
                except ValidationError as e:
                    last_err = e
            if matches != 1:
                if matches == 0 and last_err is not None:
                    raise ValidationError(path, f"oneOf: no branch matched ({last_err.message})")
                raise ValidationError(path, f"oneOf: expected exactly one match, got {matches}")
        elif combinator == "allOf":
            for sub in schema["allOf"]:
                if "if" in sub:
                    try:
                        validate(instance, sub["if"], root_schema=root_schema, root_path=root_path, path=path)
                        if_matched = True
                    except ValidationError:
                        if_matched = False
                    if if_matched and "then" in sub:
                        validate(instance, sub["then"], root_schema=root_schema, root_path=root_path, path=path)
                    if not if_matched and "else" in sub:
                        validate(instance, sub["else"], root_schema=root_schema, root_path=root_path, path=path)
                elif "not" in sub:
                    try:
                        validate(instance, sub["not"], root_schema=root_schema, root_path=root_path, path=path)
                    except ValidationError:
                        pass
                    else:
                        raise ValidationError(path, "not: sub-schema unexpectedly matched")
                elif "description" in sub and "anyOf" in sub:
                    # structural allOf entries that only delegate to anyOf
                    validate(instance, {"anyOf": sub["anyOf"]}, root_schema=root_schema, root_path=root_path, path=path)
                else:
                    validate(instance, sub, root_schema=root_schema, root_path=root_path, path=path)
        elif combinator == "anyOf":
            errors = []
            for sub in schema["anyOf"]:
                try:
                    validate(instance, sub, root_schema=root_schema, root_path=root_path, path=path)
                    break
                except ValidationError as e:
                    errors.append(e)
            else:
                raise ValidationError(path, f"anyOf: no branch matched ({errors[-1].message})")

    if "if" in schema:
        try:
            validate(instance, schema["if"], root_schema=root_schema, root_path=root_path, path=path)
            if_matched = True
        except ValidationError:
            if_matched = False
        if if_matched and "then" in schema:
            validate(instance, schema["then"], root_schema=root_schema, root_path=root_path, path=path)
        if not if_matched and "else" in schema:
            validate(instance, schema["else"], root_schema=root_schema, root_path=root_path, path=path)

    if "not" in schema:
        try:
            validate(instance, schema["not"], root_schema=root_schema, root_path=root_path, path=path)
        except ValidationError:
            pass
        else:
            raise ValidationError(path, "not: sub-schema unexpectedly matched")


# --- Artifact loaders ---------------------------------------------------------


def _load_toml(path: Path) -> Any:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _validate_file(instance: Any, schema_path: Path, label: str) -> list[str]:
    schema = _load_schema(schema_path)
    try:
        validate(instance, schema, root_schema=schema, root_path=schema_path, path="$")
        return []
    except ValidationError as e:
        return [f"[{label}] {e}"]


# --- Artifact-specific validation wrappers ------------------------------------


def validate_roster() -> list[str]:
    path = SUITE_ROOT / "roster.toml"
    if not path.exists():
        return [f"[roster] missing file: {path}"]
    data = _load_toml(path)
    return _validate_file(data, SCHEMAS_DIR / "roster.schema.json", f"roster.toml")


def validate_platform_matrix() -> list[str]:
    path = SUITE_ROOT / "platform_matrix.toml"
    if not path.exists():
        return [f"[platform_matrix] missing file: {path}"]
    data = _load_toml(path)
    return _validate_file(data, SCHEMAS_DIR / "platform_matrix.schema.json", "platform_matrix.toml")


def validate_splits() -> list[str]:
    path = SUITE_ROOT / "splits.toml"
    if not path.exists():
        return [f"[splits] missing file: {path}"]
    data = _load_toml(path)
    return _validate_file(data, SCHEMAS_DIR / "splits.schema.json", "splits.toml")


def validate_tolerances() -> list[str]:
    path = SUITE_ROOT / "tolerances.toml"
    if not path.exists():
        return [f"[tolerances] missing file: {path}"]
    data = _load_toml(path)
    return _validate_file(data, SCHEMAS_DIR / "tolerances.schema.json", "tolerances.toml")


def validate_shapes(shape_files: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in shape_files:
        data = _load_toml(path)
        errors.extend(_validate_file(data, SCHEMAS_DIR / "shape.schema.json", f"shapes/{path.name}"))
    return errors


def validate_manifest_files(manifest_files: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in manifest_files:
        data = _load_json(path)
        errors.extend(_validate_file(data, SCHEMAS_DIR / "measurement_manifest.schema.json", f"manifest:{path.name}"))
    return errors


def validate_suite_results(path: Path) -> list[str]:
    data = _load_json(path)
    errors = _validate_file(data, SCHEMAS_DIR / "suite_results.schema.json", f"suite_results:{path.name}")
    if not errors:
        status_errors = _status_integrity(data)
        errors.extend(status_errors)
    return errors


def _status_integrity(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for row in data.get("rows", []):
        if row.get("status") == "correctness_failed" and "timing_ms" in row:
            errors.append(f"[suite_results status-integrity] row {row.get('slug')}: correctness_failed_but_timed")
    return errors


# L4 pipeline slugs that require end-to-end timing + component_health in suite_results.
L4_SCORED_SLUGS = frozenset({
    "prefill_then_decode_pagedkv_pipeline",
    "moe_layer_residual_norm_pipeline",
    "speculative_decode_draft_verify_pipeline",
})


def _l4_scoring_integrity(data: dict[str, Any]) -> list[str]:
    """Every L4 row in suite_results.json must carry end-to-end timing (timing_ms)
    AND per-component component_health when status is native or sampled.
    Missing either surfaces as l4_unscored / component_checkpoint_skipped."""
    errors: list[str] = []
    for row in data.get("rows", []):
        slug = row.get("slug")
        if slug not in L4_SCORED_SLUGS:
            continue
        if row.get("status") in ("unsupported", "build_failed", "correctness_failed"):
            continue
        if "timing_ms" not in row:
            errors.append(f"[suite_results l4-scoring] row {slug}: l4_unscored (missing timing_ms)")
        if "component_health" not in row or not row.get("component_health"):
            errors.append(f"[suite_results l4-scoring] row {slug}: component_checkpoint_skipped (missing component_health)")
    return errors


def validate_l4_checkpoint(path: Path) -> list[str]:
    data = _load_json(path)
    # Identify which checkpoint schema by slug prefix
    slug = data.get("slug", "")
    schema_file = SCHEMAS_DIR / "l4_checkpoints" / f"{slug}.schema.json"
    if not schema_file.exists():
        return [f"[l4 {path.name}] no checkpoint schema for slug {slug!r}"]
    return _validate_file(data, schema_file, f"l4:{path.name}")


def validate_trace_sidecar(path: Path) -> list[str]:
    """Dispatches on filename to the matching trace-budget schema.

    Three sidecars live in the trace-budget typed-reason family
    (members of `tools/cloud/trace_budgets.py`):
    `trace_volume_exceeded`, `nvbit_slowdown_budget_exceeded`, and
    `nvbit_drop_rate_exceeded`. The C-AC-C parity scanner reads
    each `_validate_file(..., SCHEMAS_DIR / "<reason>.schema.json",
    f"trace_sidecar:...")` literal as a separate validator-schema
    surface entry; one call site per typed reason is required so a
    new family member's schema landing is visible to the parity
    scanner.
    """
    data = _load_json(path)
    diag = f"trace_sidecar:{path.relative_to(SUITE_ROOT)}"
    if path.name == "trace_volume_exceeded.json":
        return _validate_file(data, SCHEMAS_DIR / "trace_volume_exceeded.schema.json", f"trace_sidecar:{path.relative_to(SUITE_ROOT)}")
    if path.name == "nvbit_slowdown_budget_exceeded.json":
        return _validate_file(data, SCHEMAS_DIR / "nvbit_slowdown_budget_exceeded.schema.json", f"trace_sidecar:{path.relative_to(SUITE_ROOT)}")
    if path.name == "nvbit_drop_rate_exceeded.json":
        return _validate_file(data, SCHEMAS_DIR / "nvbit_drop_rate_exceeded.schema.json", f"trace_sidecar:{path.relative_to(SUITE_ROOT)}")
    return [
        f"trace_sidecar_unknown_filename:{diag}: {path.name!r} is not a "
        f"member of the trace-budget typed-reason family"
    ]


def validate_trace_convergence_confidence(path: Path) -> list[str]:
    """Validate one `trace_convergence_confidence.json` bundle (the
    adaptive-sampler clause's per-launch sidecar) against
    `workloads/simbench_bw/schemas/trace_convergence_confidence.schema.json`.

    The trace_convergence typed-reason family is declared at four
    surfaces per the C-AC-C parity inventory: producer constant
    `TRACE_CONVERGENCE_TYPED_REASONS` in
    `tools/adaptive_sampler/trace_convergence_ledger.py`; writer
    literal in `workloads/simbench_bw/run_suite.py` (the
    `_emit_trace_convergence_confidence_for_run_summary` helper);
    validator discovery at this `_validate_file(...,
    SCHEMAS_DIR / "trace_convergence_confidence.schema.json", ...)`
    call site; and the schema's admission `oneOf` typed-unavailable
    arms enumerating the same four reasons (one plan-enumerated +
    three round-added).
    """
    data = _load_json(path)
    return _validate_file(
        data,
        SCHEMAS_DIR / "trace_convergence_confidence.schema.json",
        f"trace_convergence_confidence:{path.relative_to(SUITE_ROOT)}",
    )


def _path_for_diag(path: Path) -> Path:
    """Format a path for diagnostic messages: relative to SUITE_ROOT
    when possible (the common case for in-tree artifacts under
    `results/`), otherwise the absolute path (for one-off / test-
    fixture paths under tmp / external dirs)."""
    try:
        return path.relative_to(SUITE_ROOT)
    except ValueError:
        return path


CUPTI_SIDECAR_FILENAME_RE = re.compile(r"^cupti_activity_\d+\.json$")
CUPTI_SIDECAR_SCHEMA_PATH = (
    SCHEMAS_DIR / "historical" / "cupti_activity_sidecar.v1.schema.json"
)


def is_cupti_activity_sidecar_filename(name: str) -> bool:
    """Return True iff the basename matches the CUPTI Activity sidecar
    naming convention `cupti_activity_<repeat>.json`."""
    return CUPTI_SIDECAR_FILENAME_RE.match(name) is not None


def _check_cupti_sidecar_schema_drift(data: Any, path: Path) -> list[str]:
    """Detect top-level fields in the sidecar that are NOT declared by
    schema_version 1's `properties`. Such drift is an explicit
    plan-required failure mode (typed
    `cupti_sidecar_schema_drift_without_version_bump`) — separate from
    the generic schema-violation message for missing / wrong-typed
    fields. The typed code is what reviewers grep for when investigating
    sidecars whose shape has silently widened."""
    if not isinstance(data, dict):
        return []
    schema = _load_schema(CUPTI_SIDECAR_SCHEMA_PATH)
    declared_props = set(schema.get("properties", {}).keys())
    actual_props = set(data.keys())
    extra = sorted(actual_props - declared_props)
    if not extra:
        return []
    schema_version = data.get("schema_version", "<unknown>")
    return [
        f"[cupti_sidecar_schema_drift_without_version_bump:{_path_for_diag(path)}] "
        f"sidecar contains top-level field(s) not declared in "
        f"schema_version {schema_version}: {extra}. Bumping the spec "
        f"version is required before adding new top-level fields."
    ]


# Historical dual-source reconciliation tolerance retained for replay of frozen
# CUPTI/cudaEvent evidence. The canonical bound below is enforced by this
# validator and records the contract used by the now-retired collector and
# calibration helper; no living producer is invoked here.
#
#   effective_floor_ns = max(tolerance_ns_calibrated_or_5000, 5000)
#   effective_bound_ns = max(0.05 * cuda_event_wall_time_ms * 1e6,
#                            effective_floor_ns)
#
# The calibrated value (when present in `golden/cupti_dual_source_tolerance.json`)
# is the calibrated absolute FLOOR — it can only RAISE the floor above
# the spec-default 5000 ns. The relative branch is always honored, so
# longer-duration kernels see the looser 5%-of-duration bound.
# Hand-edited fixtures with calibrated < 5000 are clamped to 5000 by
# the consumer-side `max(..., 5000)`. The formula is unified across
# all three implementations to prevent contract drift between the
# frozen producer verdict and the validator's replay recomputation.
_DUAL_SOURCE_RELATIVE_TOLERANCE = 0.05
_DUAL_SOURCE_ABSOLUTE_FLOOR_NS = 5000

CUPTI_DUAL_SOURCE_TOLERANCE_FIXTURE = (
    SUITE_ROOT / "golden" / "cupti_dual_source_tolerance.json"
)

# CUPTI Activity kinds that provide kernel-timestamp coverage for the
# dual-source primary observable. These two kinds are MUTUALLY EXCLUSIVE
# at the CUPTI API level (enabling CONCURRENT_KERNEL disables KERNEL
# and vice versa; the second cuptiActivityEnable for the conflicting
# kind returns CUPTI_ERROR_NOT_COMPATIBLE), so the retired collector enabled
# whichever one the driver accepted and a sidecar carrying EITHER kind
# in `enabled_activity_kinds` satisfies the kernel-timestamp coverage
# requirement. The historical validator check is "at least one of these two",
# not "both".
_KERNEL_TIMESTAMP_ACTIVITY_KINDS = ("_KERNEL", "_CONCURRENT_KERNEL")


_CALIBRATED_FLOOR_CACHE: dict[tuple[str, str | None], int] = {}


def _load_calibrated_tolerance_floor_ns(
    *, current_platform: str | None = None
) -> int:
    """Return the calibrated dual-source absolute floor in ns. Reads
    `golden/cupti_dual_source_tolerance.json`; when the fixture is
    missing, malformed, or carries `tolerance_ns_calibrated: null`,
    falls back to the spec-default `_DUAL_SOURCE_ABSOLUTE_FLOOR_NS`
    (5000 ns). The returned value is ALWAYS >= 5000 — a hand-edited
    fixture with a smaller value is clamped here so the consumer-side
    `max(...)` cannot tighten the runtime bound below the spec.

    The platform gate mirrors the retired producer's
    `_load_calibrated_tolerance_ns` rule: when
    `current_platform` is provided AND the fixture's
    `calibration_target_platform` does not match, the calibrated value
    is NOT honored and the spec-default 5000 ns floor is returned. This
    keeps frozen producer evidence and replay validation on the same bound for
    cross-arch captures (e.g. sm_120 captures recomputed against an
    sm_86-calibrated fixture would otherwise see a tighter floor on
    replay than the frozen producer emitted under). Callers that
    do not have access to a platform identifier (legacy callers, tests)
    pass `current_platform=None` and get the legacy unconditional
    behavior.

    Module-level cache keyed on `(fixture_path, current_platform)` so
    repeated sidecar validations do not re-read the JSON. Tests can
    clear the cache via `_clear_calibrated_floor_cache()` between
    fixtures."""
    key = (str(CUPTI_DUAL_SOURCE_TOLERANCE_FIXTURE), current_platform)
    if key in _CALIBRATED_FLOOR_CACHE:
        return _CALIBRATED_FLOOR_CACHE[key]
    floor = _DUAL_SOURCE_ABSOLUTE_FLOOR_NS
    if CUPTI_DUAL_SOURCE_TOLERANCE_FIXTURE.is_file():
        try:
            data = json.loads(CUPTI_DUAL_SOURCE_TOLERANCE_FIXTURE.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            target = data.get("calibration_target_platform")
            # Strict mirror of the retired producer's
            # `_load_calibrated_tolerance_ns`, which
            # returns -1 whenever `data.get("calibration_target_platform")
            # != current_platform`. When `current_platform` is a real
            # string and `target` is missing, `None != current_platform`
            # is True on the producer side so the calibrated value is
            # NOT honored; the validator MUST agree, otherwise older or
            # hand-edited fixtures lacking `calibration_target_platform`
            # cause producer/validator asymmetry. Legacy callers that
            # pass `current_platform=None` get the unconditional pre-fix
            # behavior so this remains backward-compatible.
            platform_match = (
                current_platform is None
                or (target is not None and target == current_platform)
            )
            value = data.get("tolerance_ns_calibrated")
            if (
                platform_match
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ):
                floor = max(value, _DUAL_SOURCE_ABSOLUTE_FLOOR_NS)
    _CALIBRATED_FLOOR_CACHE[key] = floor
    return floor


def _clear_calibrated_floor_cache() -> None:
    """Test-only: clear the cache so a per-test fixture write is
    picked up on the next `_load_calibrated_tolerance_floor_ns()`
    call. Production callers should never need this."""
    _CALIBRATED_FLOOR_CACHE.clear()


def _discover_platform_from_sidecar_dir(sidecar_path: Path) -> str | None:
    """Return the `gpu_sku` recorded in the per-shape
    `measurement_manifest.json` co-located with `sidecar_path` (the
    canonical layout writes both into the same artifact directory).
    Returns None when the manifest is missing or its `gpu_sku` is not
    a non-empty string — both cases route the platform-gate check at
    `_load_calibrated_tolerance_floor_ns` to the conservative
    legacy-unconditional branch."""
    manifest_path = sidecar_path.parent / "measurement_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    gpu_sku = data.get("gpu_sku")
    if isinstance(gpu_sku, str) and gpu_sku.strip():
        return gpu_sku
    return None


def _check_cupti_sidecar_semantics(data: Any, path: Path) -> list[str]:
    """Post-schema semantic checks on a CUPTI Activity sidecar. These
    catch the violations that schema constraints alone cannot express:
    timestamp arithmetic, dual-source agreement against the cudaEvent
    measurement, and the required activity-kind coverage. Each check
    emits a distinct typed prefix so the operator can grep for the
    specific failure mode.

    The caller MUST only invoke this after schema validation has
    passed; semantic checks assume the relevant fields are present
    and well-typed.
    """
    if not isinstance(data, dict):
        return []
    errors: list[str] = []
    diag_path = _path_for_diag(path)

    start_ns = data.get("start_ns")
    end_ns = data.get("end_ns")
    duration_ns = data.get("duration_ns")
    if isinstance(start_ns, int) and isinstance(end_ns, int):
        if end_ns < start_ns:
            errors.append(
                f"[cupti_timestamp_ordering_violated:{diag_path}] "
                f"end_ns ({end_ns}) < start_ns ({start_ns}); kernel "
                f"end timestamp must be at or after the start timestamp."
            )
        elif isinstance(duration_ns, int):
            expected = end_ns - start_ns
            if duration_ns != expected:
                errors.append(
                    f"[cupti_duration_arithmetic_violated:{diag_path}] "
                    f"duration_ns ({duration_ns}) != end_ns - start_ns "
                    f"({end_ns} - {start_ns} = {expected}); the sidecar's "
                    f"duration field must be derived from the timestamp pair."
                )

    gap_ns = data.get("dual_source_reconciliation_gap_ns")
    passed = data.get("dual_source_reconciliation_passed")
    cuda_event_ms = data.get("cuda_event_wall_time_ms")
    if (
        isinstance(gap_ns, int)
        and isinstance(cuda_event_ms, (int, float))
        and not isinstance(cuda_event_ms, bool)
        and isinstance(duration_ns, int)
    ):
        # The stored gap MUST be the signed delta between the CUPTI
        # Activity duration_ns and the cudaEvent wall-time converted
        # to ns. Validating arithmetic here (rather than only the
        # boolean-vs-bound verdict below) catches historical sidecars that
        # write a fabricated or stale gap that happens to fall inside
        # the tolerance bound — without this check, the boolean
        # verdict would still agree and the violation would slip past.
        expected_gap_ns = duration_ns - round(cuda_event_ms * 1e6)
        if gap_ns != expected_gap_ns:
            errors.append(
                f"[cupti_dual_source_gap_arithmetic_violated:{diag_path}] "
                f"dual_source_reconciliation_gap_ns ({gap_ns}) != "
                f"duration_ns - round(cuda_event_wall_time_ms * 1e6) "
                f"({duration_ns} - round({cuda_event_ms} * 1e6) = "
                f"{expected_gap_ns}); the stored gap must be the signed "
                f"delta of the two primary observables."
            )
    if (
        isinstance(gap_ns, int)
        and isinstance(passed, bool)
        and isinstance(cuda_event_ms, (int, float))
        and not isinstance(cuda_event_ms, bool)
    ):
        # Canonical bound formula — kept identical with
        # the retired collector's per-launch verdict, so the frozen
        # `dual_source_reconciliation_passed` agrees with this replay check on
        # every sidecar. The calibrated
        # floor (when the fixture is populated) only RAISES the
        # absolute branch above 5000 ns; the relative branch is
        # always honored.
        effective_floor_ns = _load_calibrated_tolerance_floor_ns(
            current_platform=_discover_platform_from_sidecar_dir(path),
        )
        bound_ns = max(
            _DUAL_SOURCE_RELATIVE_TOLERANCE * cuda_event_ms * 1e6,
            float(effective_floor_ns),
        )
        within_bound = abs(gap_ns) <= bound_ns
        if within_bound != passed:
            errors.append(
                f"[cupti_dual_source_inconsistent:{diag_path}] "
                f"dual_source_reconciliation_passed={passed} disagrees "
                f"with the recomputed verdict (|gap_ns|={abs(gap_ns)} vs "
                f"bound={bound_ns:.1f} ns derived from "
                f"max({_DUAL_SOURCE_RELATIVE_TOLERANCE} * "
                f"cuda_event_wall_time_ms={cuda_event_ms} * 1e6, "
                f"{_DUAL_SOURCE_ABSOLUTE_FLOOR_NS})); the boolean must "
                f"match the gap-vs-bound comparison exactly."
            )

    enabled_kinds = data.get("enabled_activity_kinds")
    if isinstance(enabled_kinds, list):
        present = [
            k for k in _KERNEL_TIMESTAMP_ACTIVITY_KINDS if k in enabled_kinds
        ]
        if not present:
            errors.append(
                f"[cupti_kernel_timestamp_activity_kind_missing:{diag_path}] "
                f"enabled_activity_kinds {enabled_kinds} contains neither "
                f"of the kernel-timestamp Activity kinds "
                f"{list(_KERNEL_TIMESTAMP_ACTIVITY_KINDS)}; the dual-source "
                f"primary observable requires kernel-timestamp coverage "
                f"from at least one of them (the two are mutually "
                f"exclusive at the CUPTI API level)."
            )

    return errors


def validate_cupti_activity_sidecar(path: Path) -> list[str]:
    """Replay-validate one frozen CUPTI Activity sidecar against the
    historical v1 schema plus post-schema semantic checks.

    The validator emits these distinct typed prefixes:

      - `cupti_sidecar_schema_drift_without_version_bump:` when the
        sidecar carries top-level fields beyond `schema_version` 1's
        declared properties. This is checked first so the operator
        sees the drift signal even if the schema also catches the
        extra fields via `additionalProperties: false`.
      - `cupti_sidecar_schema_violation:` for every other schema
        rule (missing required fields, wrong types, enum violations,
        etc.).
      - `cupti_timestamp_ordering_violated:` when end_ns < start_ns.
      - `cupti_duration_arithmetic_violated:` when duration_ns is not
        end_ns - start_ns.
      - `cupti_dual_source_gap_arithmetic_violated:` when the stored
        dual_source_reconciliation_gap_ns is not the signed delta
        duration_ns - round(cuda_event_wall_time_ms * 1e6). This check
        runs BEFORE the boolean-vs-bound verdict so the operator sees
        the upstream invariant first; without it, a fabricated gap
        that happens to fall inside the tolerance would slip past
        because the boolean would still agree.
      - `cupti_dual_source_inconsistent:` when the boolean
        dual_source_reconciliation_passed disagrees with the recomputed
        gap-vs-bound verdict.
      - `cupti_kernel_timestamp_activity_kind_missing:` when
        enabled_activity_kinds contains neither `_KERNEL` nor
        `_CONCURRENT_KERNEL` (the two kernel-timestamp Activity kinds
        are mutually exclusive at the CUPTI API level so the contract
        is "at least one of", not "both").

    Semantic checks are only run when the schema check passes, since
    they assume the relevant fields are present and well-typed.
    """
    data = _load_json(path)
    schema_path = CUPTI_SIDECAR_SCHEMA_PATH
    drift_errors = _check_cupti_sidecar_schema_drift(data, path)
    schema_errors = _validate_file(
        data,
        schema_path,
        f"cupti_sidecar_schema_violation:{_path_for_diag(path)}",
    )
    if drift_errors or schema_errors:
        return drift_errors + schema_errors
    semantic_errors = _check_cupti_sidecar_semantics(data, path)
    return semantic_errors


def _check_cupti_sidecars_for_manifest(
    manifest_path: Path, manifest: dict[str, Any]
) -> list[str]:
    """For one measurement_manifest.json, perform CUPTI sidecar discovery:

      - When the manifest's `enable_cupti_activity` field is true, walk
        the sibling tree for `cupti_activity_<repeat_index>.json` files
        and surface any missing as `cupti_sidecar_expected_but_missing`,
        naming the specific repeat. Validate each present sidecar
        against the schema (typed `cupti_sidecar_schema_violation`).
      - When the field is false or absent, the absence of CUPTI
        sidecars is silent (no warning, no error).
    """
    if not bool(manifest.get("enable_cupti_activity", False)):
        return []
    errors: list[str] = []
    repeat_iters = manifest.get("repeat_iters")
    if not isinstance(repeat_iters, int) or repeat_iters < 1:
        errors.append(
            f"[cupti-sidecar-discovery] manifest at "
            f"{_path_for_diag(manifest_path)} declares "
            f"enable_cupti_activity=true but has invalid repeat_iters="
            f"{repeat_iters!r}; cannot determine expected sidecar count"
        )
        return errors
    parent = manifest_path.parent
    for repeat_index in range(repeat_iters):
        sidecar_path = parent / f"cupti_activity_{repeat_index}.json"
        if not sidecar_path.is_file():
            errors.append(
                f"[cupti-sidecar-discovery] cupti_sidecar_expected_but_missing: "
                f"{_path_for_diag(sidecar_path)} (manifest at "
                f"{_path_for_diag(manifest_path)} declares "
                f"enable_cupti_activity=true and repeat_iters={repeat_iters})"
            )
            continue
        errors.extend(validate_cupti_activity_sidecar(sidecar_path))
    return errors


def discover_and_validate_all() -> list[str]:
    errors: list[str] = []
    errors.extend(validate_roster())
    errors.extend(validate_platform_matrix())
    errors.extend(validate_splits())
    errors.extend(validate_tolerances())
    shape_files = [p for p in (SUITE_ROOT / "shapes").glob("*.toml")]
    errors.extend(validate_shapes(shape_files))
    results_root = SUITE_ROOT / "results"
    if results_root.exists():
        for rp in results_root.rglob("suite_results.json"):
            errors.extend(validate_suite_results(rp))
        for mp in results_root.rglob("measurement_manifest.json"):
            # Skip dry-run manifests: the dry-run + component-strict
            # path in `run_suite.py` (line ~583, `dry_run_dir = out_dir /
            # "dry_run"`) emits a sibling measurement_manifest carrying
            # synthetic identity fields from `synthetic_dry_run_props()`
            # (sm_count=0, device_name="DRY-RUN-CPU", gpu_uuid="GPU-dry-
            # run-cpu-..."). The schema requires sm_count >= 1 (real
            # measurements only), so validating the dry-run artifact would
            # always fail. The `dry_run/` parent directory is the durable
            # contract for distinguishing dry-run scaffolds from real
            # measurements.
            if mp.parent.name == "dry_run":
                continue
            errors.extend(validate_manifest_files([mp]))
            # CUPTI sidecar discovery: only manifests declaring
            # enable_cupti_activity=true expect sibling cupti_activity_*.json
            # files. Walk per-manifest so the "silent when CUPTI not
            # enabled" path requires no opt-in flag.
            try:
                manifest_data = _load_json(mp)
            except Exception:
                continue  # surface schema errors via validate_manifest_files above
            if isinstance(manifest_data, dict):
                errors.extend(_check_cupti_sidecars_for_manifest(mp, manifest_data))
        for cp in results_root.rglob("l4_checkpoint_*.json"):
            errors.extend(validate_l4_checkpoint(cp))
        for sp in results_root.rglob("trace_volume_exceeded.json"):
            errors.extend(validate_trace_sidecar(sp))
        for sp in results_root.rglob("nvbit_slowdown_budget_exceeded.json"):
            errors.extend(validate_trace_sidecar(sp))
        for sp in results_root.rglob("nvbit_drop_rate_exceeded.json"):
            errors.extend(validate_trace_sidecar(sp))
        for sp in results_root.rglob("convergence_confidence.json"):
            errors.extend(validate_trace_convergence_confidence(sp))
    return errors


# --- CLI ----------------------------------------------------------------------


MEASUREMENT_MANIFEST_BASENAMES = ("manifest.json", "measurement_manifest.json")


def _check_measurement_manifest_for_suite_results(
    suite_results_path: Path,
) -> list[str]:
    """Check the manifest-presence invariant for one suite_results.json.

    For every row with ``status="native"`` or ``status="sampled"``
    (the schema's timing-result arms that require
    ``measurement_manifest_path``), resolve the recorded manifest path
    relative to the suite_results.json's parent directory and check it
    exists.

    Falls back to checking ``parent / "measurement_manifest.json"``
    for rows that don't carry the field (e.g., rows produced by
    older code paths that didn't record the path explicitly) so
    pre-existing artifacts continue to admit.

    Returns a list of ``measurement_manifest_missing: <path>``
    diagnostics; empty list when every native/sampled row's manifest
    is present at the recorded location.
    """
    parent = suite_results_path.parent
    try:
        data = _load_json(suite_results_path)
    except Exception as exc:
        return [
            f"[manifest-presence] suite_results_unreadable: "
            f"{suite_results_path} ({exc})"
        ]
    if not isinstance(data, dict):
        return []
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        # No rows → no manifest expectation. Schema-level checks
        # are handled elsewhere via validate_suite_results.
        return []
    errors: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("status") not in {"native", "sampled"}:
            # Non-timing statuses (unsupported, build/correctness
            # failures, etc.) are not required to carry a measurement
            # manifest.
            continue
        relpath = row.get("measurement_manifest_path")
        if not isinstance(relpath, str) or not relpath:
            # Backward-compat: rows without the field fall back to
            # the legacy parent-level location.
            relpath = "measurement_manifest.json"
        manifest_path = parent / relpath
        if not manifest_path.is_file():
            errors.append(
                f"[manifest-presence] measurement_manifest_missing: "
                f"{suite_results_path} (expected at "
                f"{relpath!r} relative to row dir; resolved "
                f"{manifest_path})"
            )
    return errors


def _run_schema_check(target: Path) -> int:
    """Implement the --schema-check PATH CLI surface.

    Behaviour:
      - PATH does not exist -> exit 2 (`schema_check_path_missing`).
      - PATH is a file with an unsupported basename -> exit 2
        (`schema_check_unsupported_file`).
      - PATH is a directory containing no manifest files -> exit 2
        (`schema_check_no_manifests`).
      - One or more manifests found and any fails the schema -> exit 1
        with a `schema_validation_error:` line per failure.
      - One or more manifests found and all validate -> exit 0 with `ok`.
    """
    if not target.exists():
        print(
            f"schema_check_path_missing: {target}",
            file=sys.stderr,
        )
        return 2
    if target.is_file() and target.name not in MEASUREMENT_MANIFEST_BASENAMES:
        print(
            f"schema_check_unsupported_file: {target} "
            f"(expected basename one of {list(MEASUREMENT_MANIFEST_BASENAMES)})",
            file=sys.stderr,
        )
        return 2
    manifests = discover_manifests(target)
    if not manifests:
        print(
            f"schema_check_no_manifests: {target} "
            f"(no manifest.json or measurement_manifest.json files found)",
            file=sys.stderr,
        )
        return 2
    errors = validate_manifest_files(manifests)
    if errors:
        for e in errors:
            print(f"schema_validation_error: {e}", file=sys.stderr)
        print(
            f"schema_check_failed: {len(errors)} error(s) across "
            f"{len(manifests)} manifest(s)",
            file=sys.stderr,
        )
        return 1
    print(f"ok ({len(manifests)} manifest(s) validated)")
    return 0


def discover_manifests(target: Path) -> list[Path]:
    """Return measurement manifests under `target`. When `target` is a
    file, returns `[target]`. When `target` is a directory, returns every
    file whose basename is `manifest.json` or `measurement_manifest.json`,
    sorted for deterministic output. The caller is responsible for
    deciding what to do with an empty result."""
    if target.is_file():
        return [target]
    manifests: list[Path] = []
    for name in MEASUREMENT_MANIFEST_BASENAMES:
        manifests.extend(target.rglob(name))
    return sorted(set(manifests))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SimBench-BW artifact schema validator")
    p.add_argument("--all", action="store_true", help="Validate every discoverable artifact")
    p.add_argument("--measurement-manifest-present", action="store_true",
                   help="Assert that every result file has a sibling measurement_manifest.json")
    p.add_argument("--status-integrity", action="store_true",
                   help="Assert suite_results.json status/timing consistency")
    p.add_argument("--check-l4-scoring", action="store_true",
                   help="Assert every L4 row has end-to-end timing + component_health in suite_results.json")
    p.add_argument("--schema-check", metavar="PATH",
                   help="Validate every measurement manifest under PATH against "
                        "measurement_manifest.schema.json. PATH may be a single "
                        "manifest.json/measurement_manifest.json file or a directory "
                        "(walked recursively). Exits 0 when at least one manifest "
                        "is found and all validate, 1 with diagnostics on schema "
                        "failures, 2 on missing path / empty directory / unsupported file.")
    p.add_argument("artifact", nargs="?", help="Validate a specific artifact file")
    args = p.parse_args(argv)

    if args.schema_check is not None:
        return _run_schema_check(Path(args.schema_check))

    selected = args.all or args.measurement_manifest_present or args.status_integrity or args.check_l4_scoring or args.artifact
    if not selected:
        p.error("specify --all or an artifact path")  # exits with code 2 inside argparse

    errors: list[str] = []
    if args.all:
        errors.extend(discover_and_validate_all())
    if args.artifact:
        ap = Path(args.artifact)
        if not ap.exists():
            print(f"missing: {ap}", file=sys.stderr)
            return 2
        if ap.suffix == ".toml":
            data = _load_toml(ap)
        else:
            data = _load_json(ap)
        # Try to infer which schema by filename
        inferred = {
            "roster.toml": SCHEMAS_DIR / "roster.schema.json",
            "platform_matrix.toml": SCHEMAS_DIR / "platform_matrix.schema.json",
            "splits.toml": SCHEMAS_DIR / "splits.schema.json",
            "tolerances.toml": SCHEMAS_DIR / "tolerances.schema.json",
        }.get(ap.name)
        if inferred is None and ap.parent.name == "shapes":
            inferred = SCHEMAS_DIR / "shape.schema.json"
        if inferred is None and is_cupti_activity_sidecar_filename(ap.name):
            # CUPTI Activity sidecar: route through the wrapper that
            # also emits the typed schema-drift code for extra
            # top-level fields. Returning here bypasses the generic
            # _validate_file call below so the drift-aware error path
            # is the one the user sees.
            errors.extend(validate_cupti_activity_sidecar(ap))
            # The wrapper returns below on both success and failure. Keep the
            # inferred value truthful for the shared guard in this branch.
            inferred = CUPTI_SIDECAR_SCHEMA_PATH
            if errors:
                for e in errors:
                    print(e, file=sys.stderr)
                print(f"{len(errors)} validation error(s)", file=sys.stderr)
                return 1
            print("ok")
            return 0
        if inferred is None:
            print(f"no inferable schema for {ap.name}; use --all", file=sys.stderr)
            return 2
        errors.extend(_validate_file(data, inferred, ap.name))
    if args.status_integrity:
        results_root = SUITE_ROOT / "results"
        if results_root.exists():
            for rp in results_root.rglob("suite_results.json"):
                data = _load_json(rp)
                errors.extend(_status_integrity(data))
    if args.check_l4_scoring:
        results_root = SUITE_ROOT / "results"
        any_found = False
        if results_root.exists():
            for rp in results_root.rglob("suite_results.json"):
                any_found = True
                data = _load_json(rp)
                errors.extend(_l4_scoring_integrity(data))
        if not any_found:
            # No suite_results.json yet is not a failure — the check returns ok
            # only if nothing to validate. We explicitly do not fail here so pre-
            # kernel-landing rounds can still run the flag as a sanity pass.
            pass
    if args.measurement_manifest_present:
        results_root = SUITE_ROOT / "results"
        if results_root.exists():
            for rp in results_root.rglob("suite_results.json"):
                errors.extend(
                    _check_measurement_manifest_for_suite_results(rp),
                )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"{len(errors)} validation error(s)", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
