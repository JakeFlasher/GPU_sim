"""Importer for local-label sidecars.

Reads a per-launch JSON sidecar produced by tools/local_labels/label_driver.sh
on local hardware (RTX 3090 Ti sm_86 or RTX 5060 laptop sm_120), runs JSON
Schema validation, enforces cross-field constraints (the confidence
interval's ``lower <= upper``), and rejects any document whose JSON tree
contains a field name from the forbidden denylist (raw HW-counter event
identifiers and similar surfaces that the non-root contract forbids on
cloud GPUs).

Sidecar paths follow ``data/local_labels/<device>/<slug>/<shape_point>/label_<repeat>_<launch>.json``.

Public entry points:

- ``load_schema()`` returns the bundled JSON Schema document as a dict.
- ``validate(sidecar)`` runs all checks against an in-memory parsed dict.
- ``validate_path(path)`` opens the file and forwards to ``validate``.
- ``main(argv)`` is the CLI entry point; exits non-zero on any failure.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import jsonschema

# Package-safe sibling import (the module-tree convention, mirroring
# emit_ac4_load_bearing_coverage.py): `tools/local_labels` is a
# namespace package (no __init__.py), so callers using the documented
# package path (`from tools.local_labels import import_label` with only
# the repo root on sys.path) would otherwise hit ModuleNotFoundError on
# the bare sibling import below. Inserting the module dir first keeps
# BOTH conventions working: package-path imports and the local
# script/test convention that runs with tools/local_labels prepended.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metric_registry import load_registry, validate_unit_tag  # noqa: E402
from ncu_capture_contract import validate_measurement_protocol  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "local_label_sidecar.schema.json"
)


# Field-name tokens that imply raw HW-counter event identification or
# privileged CUPTI Profiling surfaces. Embedding any of these names in a
# label sidecar would let a downstream consumer re-encode admin-required
# signal back into the data path. The non-root contract forbids those
# surfaces on cloud GPUs and the label predictors themselves run on
# cloud GPUs, so the labels must not carry such identifiers either.
_FORBIDDEN_FIELD_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Raw HW counter classes.
    re.compile(r"^hw_counter(?:_|$)"),
    re.compile(r"^raw_hw_counter(?:_|$)"),
    re.compile(r"^pm_counter(?:_|$)"),
    re.compile(r"^pm_sample(?:_|$)"),
    re.compile(r"^pm_sampling(?:_|$)"),
    re.compile(r"^counter_id(?:_|$)"),
    re.compile(r"^counter_name(?:_|$)"),
    # CUPTI Profiling surfaces.
    re.compile(r"^cupti_event(?:_|$)"),
    re.compile(r"^cupti_metric(?:_|$)"),
    re.compile(r"^cupti_pc_sampling(?:_|$)"),
    re.compile(r"^range_profiler(?:_|$)"),
    re.compile(r"^sass_metric(?:_|$)"),
    re.compile(r"^pc_sampling(?:_|$)"),
    re.compile(r"^instruction_execution(?:_|$)"),
    re.compile(r"^instruction_correlation(?:_|$)"),
    # Generic event / metric / signal identifiers (cover bare-name
    # variants the more specific patterns above might miss).
    re.compile(r"^event_id(?:_|$)"),
    re.compile(r"^event_name(?:_|$)"),
    re.compile(r"^metric_id(?:_|$)"),
    re.compile(r"^metric_name(?:_|$)"),
    re.compile(r"^signal_id(?:_|$)"),
    re.compile(r"^signal_name(?:_|$)"),
    # NVIDIA performance-tooling surfaces.
    re.compile(r"^nvperf(?:_|$)"),
    re.compile(r"^perfworks(?:_|$)"),
)


@dataclass
class ValidationResult:
    """Outcome of importing a local-label sidecar.

    ``ok`` is ``True`` only when every check passed. On failure
    ``typed_reason`` carries one of the contract's typed exclusion codes
    (``forbidden_field_name``, ``schema_validation_failed``,
    ``confidence_interval_inverted``, ``sidecar_unreadable``) and
    ``detail`` carries a human-readable explanation. ``schema_errors``
    is populated only on schema-validation failures and lists every
    violation the validator surfaced (so the caller can see all errors,
    not just the first ``5`` shown in ``detail``).
    """

    ok: bool
    typed_reason: str = ""
    detail: str = ""
    schema_errors: list[str] = field(default_factory=list)


def load_schema() -> dict:
    """Load the bundled JSON Schema document for local-label sidecars."""
    with _SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _walk_field_names(obj: Any) -> Iterable[tuple[list[str], str]]:
    """Yield ``(json_path_components, field_name)`` for every field name
    in a JSON tree. Recurses into nested objects and arrays so a
    forbidden name buried under ``provenance.extra.metadata.event_id``
    is still caught."""
    if isinstance(obj, dict):
        for name, value in obj.items():
            yield ([], name)
            for path, inner in _walk_field_names(value):
                yield ([name] + path, inner)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            for path, inner in _walk_field_names(value):
                yield ([f"[{index}]"] + path, inner)


def _check_forbidden_field_names(sidecar: Any) -> ValidationResult | None:
    for path_components, name in _walk_field_names(sidecar):
        for pattern in _FORBIDDEN_FIELD_NAME_PATTERNS:
            if pattern.search(name):
                full_path = (
                    ".".join(path_components + [name])
                    if path_components
                    else name
                )
                return ValidationResult(
                    ok=False,
                    typed_reason="forbidden_field_name",
                    detail=(
                        f"Label sidecar contains forbidden field name "
                        f"'{name}' at JSON path '{full_path}'. The "
                        f"non-root contract forbids embedding raw "
                        f"HW-counter event identifiers or privileged "
                        f"CUPTI Profiling surfaces in label sidecars."
                    ),
                )
    return None


def _check_confidence_interval_ordering(
    sidecar: dict,
) -> ValidationResult | None:
    for ci_field in (
        "confidence_interval_l2_sector",
        "confidence_interval_l1tex_sector",
    ):
        ci = sidecar.get(ci_field)
        if ci is None:
            continue
        try:
            lower = ci["lower"]
            upper = ci["upper"]
        except (KeyError, TypeError):
            # Schema validation already flagged this; defensive skip so
            # we surface the first cross-field violation cleanly.
            continue
        if lower > upper:
            return ValidationResult(
                ok=False,
                typed_reason="confidence_interval_inverted",
                detail=(
                    f"'{ci_field}.lower' ({lower}) is greater than "
                    f"'{ci_field}.upper' ({upper}). The interval must "
                    f"satisfy 'lower <= upper'."
                ),
            )
    return None


def _check_unit_tagged_metrics(sidecar: dict) -> ValidationResult | None:
    """Cross-check the optional v5 ``unit_tagged_metrics`` block against the
    committed metric registry.

    Each record's ``(ncu_metric, unit)`` must resolve to a registry entry whose
    unit matches — this is the record-layer guard that a ``count`` / ``ns`` /
    ``cycles`` value can never be smuggled in under the percentage unit (or vice
    versa). The optional per-metric ``confidence_interval`` must satisfy
    ``lower <= upper`` and carry the same unit as its parent record.
    """
    records = sidecar.get("unit_tagged_metrics")
    if not records:
        return None
    registry = load_registry()
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue  # schema validation already rejected this shape
        ncu_metric = rec.get("ncu_metric")
        unit = rec.get("unit")
        ok, typed_reason, detail = validate_unit_tag(registry, ncu_metric, unit)
        if not ok:
            return ValidationResult(
                ok=False,
                typed_reason=typed_reason,
                detail=f"unit_tagged_metrics[{idx}]: {detail}",
            )
        # validate_unit_tag passed, so the entry resolves. Enforce the
        # registry's declared valid range on the value (and any CI bound):
        # e.g. a pct metric carries a fraction in [0, 1], so a value of 999.0
        # must be rejected even though the JSON schema only pins minimum 0.
        entry = registry.by_ncu_metric(ncu_metric)
        range_violation = _check_value_in_registry_range(
            entry, rec.get("value"), idx, "value"
        )
        if range_violation is not None:
            return range_violation
        ci = rec.get("confidence_interval")
        if isinstance(ci, dict):
            lower, upper = ci.get("lower"), ci.get("upper")
            if (
                isinstance(lower, (int, float))
                and isinstance(upper, (int, float))
                and lower > upper
            ):
                return ValidationResult(
                    ok=False,
                    typed_reason="unit_tagged_confidence_interval_inverted",
                    detail=(
                        f"unit_tagged_metrics[{idx}].confidence_interval.lower "
                        f"({lower}) is greater than upper ({upper})"
                    ),
                )
            if ci.get("unit") not in (None, unit):
                return ValidationResult(
                    ok=False,
                    typed_reason="unit_tagged_metric_unit_mismatch",
                    detail=(
                        f"unit_tagged_metrics[{idx}].confidence_interval.unit "
                        f"{ci.get('unit')!r} != record unit {unit!r}"
                    ),
                )
            for bound_name in ("lower", "upper"):
                bound_violation = _check_value_in_registry_range(
                    entry,
                    ci.get(bound_name),
                    idx,
                    f"confidence_interval.{bound_name}",
                )
                if bound_violation is not None:
                    return bound_violation
    return None


def _check_value_in_registry_range(
    entry: Any, value: Any, idx: int, field: str
) -> ValidationResult | None:
    """Reject ``value < entry.valid_min`` or (when ``entry.valid_max`` is not
    None) ``value > entry.valid_max`` for a unit-tagged metric value or CI
    bound. Non-numeric values are left to JSON-Schema type validation."""
    if not isinstance(value, (int, float)):
        return None
    if value < entry.valid_min or (
        entry.valid_max is not None and value > entry.valid_max
    ):
        upper = "inf" if entry.valid_max is None else entry.valid_max
        return ValidationResult(
            ok=False,
            typed_reason="unit_tagged_metric_value_out_of_range",
            detail=(
                f"unit_tagged_metrics[{idx}].{field} {value} is outside the "
                f"registry valid range [{entry.valid_min}, {upper}] for "
                f"{entry.ncu_metric!r} (unit {entry.unit})"
            ),
        )
    return None


def validate(sidecar: Any, schema: dict | None = None) -> ValidationResult:
    """Validate a parsed local-label sidecar.

    Order of checks: forbidden-field denylist (whole JSON tree) -> JSON
    Schema -> explicit NCU protocol semantics -> confidence-interval ordering.
    The forbidden-field check
    runs first so a sidecar that smuggles in a raw HW-counter event
    identifier fails with a specific typed reason even before structural
    schema validation reaches the offending field.
    """
    forbidden = _check_forbidden_field_names(sidecar)
    if forbidden is not None:
        return forbidden

    if schema is None:
        schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(sidecar), key=lambda e: list(e.path))
    if errors:
        formatted = [
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        ]
        return ValidationResult(
            ok=False,
            typed_reason="schema_validation_failed",
            detail="; ".join(formatted[:5]),
            schema_errors=formatted,
        )

    if not isinstance(sidecar, dict):
        # Schema rejects non-objects upstream, but defend in case a
        # caller passes the schema explicitly to skip that branch.
        return ValidationResult(
            ok=False,
            typed_reason="schema_validation_failed",
            detail="sidecar root must be a JSON object",
        )

    try:
        validate_measurement_protocol(sidecar.get("measurement_protocol"))
    except ValueError as exc:
        return ValidationResult(
            ok=False,
            typed_reason="schema_validation_failed",
            detail=f"measurement_protocol: {exc}",
        )

    ordering = _check_confidence_interval_ordering(sidecar)
    if ordering is not None:
        return ordering

    unit_tagged = _check_unit_tagged_metrics(sidecar)
    if unit_tagged is not None:
        return unit_tagged

    return ValidationResult(ok=True)


def validate_path(path: Path) -> ValidationResult:
    """Open a JSON file and validate its contents."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            sidecar = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationResult(
            ok=False,
            typed_reason="sidecar_unreadable",
            detail=f"{path}: {exc}",
        )
    return validate(sidecar)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate one or more local-label sidecar JSON files against "
            "the schema, cross-field, and forbidden-field-name "
            "constraints."
        )
    )
    parser.add_argument(
        "sidecars",
        nargs="+",
        type=Path,
        help="Path(s) to per-launch sidecar JSON files.",
    )
    args = parser.parse_args(argv)

    fail = 0
    for path in args.sidecars:
        result = validate_path(path)
        if result.ok:
            print(f"OK   {path}")
        else:
            print(f"FAIL {path}: {result.typed_reason}: {result.detail}")
            fail += 1
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
