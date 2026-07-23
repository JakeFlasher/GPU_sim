"""Calibration corpus manifest loader + validator.

The manifest at `calibration_corpus.toml` contract-pins the
calibration cohort partition before any NCU sweep runs:
every cell is predeclared with one of three predictor
buckets — friendly, hostile, or typed_unavailable. Cohort A
(sm_86 microbench cells) and cohort B (SOL-ExecBench
representative cells) are kept separate so downstream
report tables preserve the cohort split (no cross-cohort
averaging into a single headline).

Public surface:

  * `CalibrationCorpusManifest` — frozen dataclass tree.
  * `CalibrationCorpusCell` — per-cell record (cohort name +
    cell_name + predictor_bucket + rationale + optional
    binary_basename / problem_slug).
  * `CalibrationCorpusMeta` — per-manifest preamble
    (schema_version, generated_at, cohort_split_policy,
    predictor_bucket_enum, typed_unavailable_subclass).
  * `CorpusManifestError(typed_reason, message)` — raised by
    the loader and validator. Typed reasons:
    `corpus_manifest_schema_violation`,
    `corpus_manifest_bucket_invalid`,
    `corpus_manifest_duplicate_cell_name`,
    `corpus_manifest_missing_field`,
    `corpus_manifest_schema_version_mismatch`,
    `corpus_manifest_io_error`.
  * `load_calibration_corpus_manifest(path)` — reads the
    TOML and returns the dataclass tree.
  * `validate_calibration_corpus_manifest(manifest)` —
    raises `CorpusManifestError` on schema or content
    violations; returns silently when clean.

Importing this module does NOT invoke NCU. The manifest is
the calibration corpus contract; the calibration sweep that
consumes it is a separate module.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CALIBRATION_CORPUS_SCHEMA_VERSION: str = "1.0"


CORPUS_MANIFEST_PREDICTOR_BUCKETS: frozenset[str] = frozenset({
    "predictor_friendly",
    "predictor_hostile",
    "predictor_typed_unavailable",
})


CORPUS_MANIFEST_TYPED_REASONS: frozenset[str] = frozenset({
    "corpus_manifest_schema_violation",
    "corpus_manifest_bucket_invalid",
    "corpus_manifest_duplicate_cell_name",
    "corpus_manifest_missing_field",
    "corpus_manifest_schema_version_mismatch",
    "corpus_manifest_io_error",
    "corpus_manifest_typed_unavailable_subclass_missing",
    "corpus_manifest_subclass_on_non_typed_unavailable_cell",
    "corpus_manifest_measured_cache_initial_state_invalid",
})


# The set of values permitted for `measured_cache_initial_state` on a
# per-cell manifest row. Empty string means "use the protocol-derived
# default from `derive_warmup_protocol`" (legacy behavior for cells
# that predate the carrier). `"warm"` and `"cold"` are the explicit
# carriers that override the protocol-derived value at the
# calibration-sweep boundary.
CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES: frozenset[str] = frozenset({
    "",
    "warm",
    "cold",
})


class CorpusManifestError(Exception):
    """Raised by the manifest loader / validator on schema
    or content violations. Carries the typed-reason string
    so downstream callers (the calibration sweep, cohort
    partition probes, the typed-unavailable factory, etc.)
    can compose typed-unavailable bundles or report-table
    errors.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class CalibrationCorpusMeta:
    schema_version: str
    generated_at: str
    cohort_split_policy: str
    predictor_bucket_enum: tuple[str, ...]
    typed_unavailable_subclass: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationCorpusCell:
    cohort: str
    cell_name: str
    predictor_bucket: str
    rationale: str
    binary_basename: str = ""
    problem_slug: str = ""
    typed_unavailable_subclass: str = ""
    shape_point: str = ""
    # Warm/cold cache-protocol carrier. Empty default means "use the
    # protocol-derived `cache_initial_state` from
    # `derive_warmup_protocol`"; an explicit `"warm"` or `"cold"`
    # overrides the derivation at the calibration-sweep boundary.
    # The closed vocabulary is `CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES`.
    measured_cache_initial_state: str = ""


@dataclass(frozen=True)
class CalibrationCorpusManifest:
    meta: CalibrationCorpusMeta
    cohort_a_description: str
    cohort_a_binary_root: str
    cohort_a_cells: tuple[CalibrationCorpusCell, ...]
    cohort_b_description: str
    cohort_b_problem_root: str
    cohort_b_cells: tuple[CalibrationCorpusCell, ...]


def load_calibration_corpus_manifest(
    path: Path,
) -> CalibrationCorpusManifest:
    """Read the TOML manifest at `path` and return the
    parsed dataclass tree. Raises `CorpusManifestError` with
    `corpus_manifest_io_error` if the file is missing or
    unreadable, and `corpus_manifest_schema_violation` /
    `corpus_manifest_missing_field` for structural problems
    the loader detects before the validator runs.
    """
    try:
        raw_bytes = Path(path).read_bytes()
    except (OSError, FileNotFoundError) as exc:
        raise CorpusManifestError(
            "corpus_manifest_io_error",
            f"cannot read manifest at {path!r}: {exc}",
        ) from exc
    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"manifest is not valid TOML: {exc}",
        ) from exc

    meta = _build_meta(data)
    cohort_a_description, cohort_a_binary_root, cohort_a_cells = (
        _build_cohort_a(data)
    )
    cohort_b_description, cohort_b_problem_root, cohort_b_cells = (
        _build_cohort_b(data)
    )
    return CalibrationCorpusManifest(
        meta=meta,
        cohort_a_description=cohort_a_description,
        cohort_a_binary_root=cohort_a_binary_root,
        cohort_a_cells=tuple(cohort_a_cells),
        cohort_b_description=cohort_b_description,
        cohort_b_problem_root=cohort_b_problem_root,
        cohort_b_cells=tuple(cohort_b_cells),
    )


def _build_meta(data: dict[str, Any]) -> CalibrationCorpusMeta:
    meta_section = _required_table(data, "meta")
    schema_version = _required_str(meta_section, "meta.schema_version")
    generated_at = _required_str(meta_section, "meta.generated_at")
    cohort_split_policy = _required_str(
        meta_section, "meta.cohort_split_policy",
    )
    bucket_enum = _required_str_list(
        meta_section, "meta.predictor_bucket_enum",
    )
    typed_unavailable_subclass = _required_str_list(
        meta_section, "meta.typed_unavailable_subclass",
    )
    return CalibrationCorpusMeta(
        schema_version=schema_version,
        generated_at=generated_at,
        cohort_split_policy=cohort_split_policy,
        predictor_bucket_enum=tuple(bucket_enum),
        typed_unavailable_subclass=tuple(typed_unavailable_subclass),
    )


def _build_cohort_a(
    data: dict[str, Any],
) -> tuple[str, str, list[CalibrationCorpusCell]]:
    section = _required_table(data, "cohort_a")
    description = _required_str(section, "cohort_a.description")
    binary_root = _required_str(section, "cohort_a.binary_root")
    raw_cells = section.get("cells", [])
    if not isinstance(raw_cells, list) or not raw_cells:
        raise CorpusManifestError(
            "corpus_manifest_missing_field",
            "cohort_a.cells must be a non-empty array of "
            "tables; got "
            f"{type(raw_cells).__name__}",
        )
    cells: list[CalibrationCorpusCell] = []
    for index, raw_cell in enumerate(raw_cells):
        cells.append(_build_cohort_a_cell(raw_cell, index))
    return description, binary_root, cells


def _build_cohort_a_cell(
    raw: Any, index: int,
) -> CalibrationCorpusCell:
    if not isinstance(raw, dict):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"cohort_a.cells[{index}] must be a table; got "
            f"{type(raw).__name__}",
        )
    cell_name = _required_str(
        raw, f"cohort_a.cells[{index}].cell_name",
    )
    binary_basename = _required_str(
        raw, f"cohort_a.cells[{index}].binary_basename",
    )
    bucket = _required_str(
        raw, f"cohort_a.cells[{index}].predictor_bucket",
    )
    rationale = _required_str(
        raw, f"cohort_a.cells[{index}].rationale",
    )
    typed_unavailable_subclass = _optional_str(
        raw, "typed_unavailable_subclass",
    )
    # shape_point defaults to cell_name when absent; the new
    # field is optional in the manifest schema for backward
    # compatibility with manifests authored before the
    # framework-contract 7-tuple landed.
    shape_point = _optional_str(raw, "shape_point") or cell_name
    measured_cache_initial_state = _optional_str(
        raw, "measured_cache_initial_state",
    )
    return CalibrationCorpusCell(
        cohort="cohort_a",
        cell_name=cell_name,
        predictor_bucket=bucket,
        rationale=rationale,
        binary_basename=binary_basename,
        typed_unavailable_subclass=typed_unavailable_subclass,
        shape_point=shape_point,
        measured_cache_initial_state=measured_cache_initial_state,
    )


def _build_cohort_b(
    data: dict[str, Any],
) -> tuple[str, str, list[CalibrationCorpusCell]]:
    section = _required_table(data, "cohort_b")
    description = _required_str(section, "cohort_b.description")
    problem_root = _required_str(section, "cohort_b.problem_root")
    raw_cells = section.get("cells", [])
    if not isinstance(raw_cells, list) or not raw_cells:
        raise CorpusManifestError(
            "corpus_manifest_missing_field",
            "cohort_b.cells must be a non-empty array of "
            "tables; got "
            f"{type(raw_cells).__name__}",
        )
    cells: list[CalibrationCorpusCell] = []
    for index, raw_cell in enumerate(raw_cells):
        cells.append(_build_cohort_b_cell(raw_cell, index))
    return description, problem_root, cells


def _build_cohort_b_cell(
    raw: Any, index: int,
) -> CalibrationCorpusCell:
    if not isinstance(raw, dict):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"cohort_b.cells[{index}] must be a table; got "
            f"{type(raw).__name__}",
        )
    cell_name = _required_str(
        raw, f"cohort_b.cells[{index}].cell_name",
    )
    problem_slug = _required_str(
        raw, f"cohort_b.cells[{index}].problem_slug",
    )
    bucket = _required_str(
        raw, f"cohort_b.cells[{index}].predictor_bucket",
    )
    rationale = _required_str(
        raw, f"cohort_b.cells[{index}].rationale",
    )
    typed_unavailable_subclass = _optional_str(
        raw, "typed_unavailable_subclass",
    )
    # shape_point defaults to cell_name when absent; cohort B
    # cells inherit the same backward-compat rule as cohort A.
    shape_point = _optional_str(raw, "shape_point") or cell_name
    measured_cache_initial_state = _optional_str(
        raw, "measured_cache_initial_state",
    )
    return CalibrationCorpusCell(
        cohort="cohort_b",
        cell_name=cell_name,
        predictor_bucket=bucket,
        rationale=rationale,
        problem_slug=problem_slug,
        typed_unavailable_subclass=typed_unavailable_subclass,
        shape_point=shape_point,
        measured_cache_initial_state=measured_cache_initial_state,
    )


def validate_calibration_corpus_manifest(
    manifest: CalibrationCorpusManifest,
) -> None:
    """Raise `CorpusManifestError` on schema or content
    violations. Returns silently when the manifest is
    clean.

    Checks:
      * `schema_version == CALIBRATION_CORPUS_SCHEMA_VERSION`.
      * Each cell's `predictor_bucket` is in
        `CORPUS_MANIFEST_PREDICTOR_BUCKETS`.
      * No duplicate `cell_name` across both cohorts.
      * Every cell has a non-empty `rationale`.
      * The `predictor_bucket_enum` declared in meta
        matches `CORPUS_MANIFEST_PREDICTOR_BUCKETS` as a
        set (order in the manifest is free).
    """
    if not isinstance(manifest, CalibrationCorpusManifest):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            "argument is not a CalibrationCorpusManifest; "
            f"got {type(manifest).__name__}",
        )
    if manifest.meta.schema_version != CALIBRATION_CORPUS_SCHEMA_VERSION:
        raise CorpusManifestError(
            "corpus_manifest_schema_version_mismatch",
            f"manifest schema_version "
            f"{manifest.meta.schema_version!r} does not "
            f"equal the pinned "
            f"{CALIBRATION_CORPUS_SCHEMA_VERSION!r}",
        )
    declared_enum = set(manifest.meta.predictor_bucket_enum)
    if declared_enum != set(CORPUS_MANIFEST_PREDICTOR_BUCKETS):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"meta.predictor_bucket_enum "
            f"{sorted(declared_enum)} does not match the "
            f"pinned bucket set "
            f"{sorted(CORPUS_MANIFEST_PREDICTOR_BUCKETS)}",
        )

    typed_unavailable_subclass_enum = set(
        manifest.meta.typed_unavailable_subclass,
    )
    seen_names: dict[str, str] = {}
    for cell in (
        *manifest.cohort_a_cells,
        *manifest.cohort_b_cells,
    ):
        if cell.predictor_bucket not in CORPUS_MANIFEST_PREDICTOR_BUCKETS:
            raise CorpusManifestError(
                "corpus_manifest_bucket_invalid",
                f"cell {cell.cell_name!r} ({cell.cohort}) "
                f"has predictor_bucket "
                f"{cell.predictor_bucket!r}; expected one "
                f"of "
                f"{sorted(CORPUS_MANIFEST_PREDICTOR_BUCKETS)}",
            )
        if not cell.rationale.strip():
            raise CorpusManifestError(
                "corpus_manifest_missing_field",
                f"cell {cell.cell_name!r} ({cell.cohort}) "
                f"has an empty rationale",
            )
        if cell.predictor_bucket == "predictor_typed_unavailable":
            if not cell.typed_unavailable_subclass.strip():
                raise CorpusManifestError(
                    "corpus_manifest_typed_unavailable_subclass_missing",
                    f"cell {cell.cell_name!r} ({cell.cohort}) "
                    f"is predictor_typed_unavailable but "
                    f"carries no typed_unavailable_subclass",
                )
            if cell.typed_unavailable_subclass not in (
                typed_unavailable_subclass_enum
            ):
                raise CorpusManifestError(
                    "corpus_manifest_typed_unavailable_subclass_missing",
                    f"cell {cell.cell_name!r} ({cell.cohort}) "
                    f"has typed_unavailable_subclass "
                    f"{cell.typed_unavailable_subclass!r}; "
                    f"expected one of "
                    f"{sorted(typed_unavailable_subclass_enum)} "
                    f"from meta.typed_unavailable_subclass",
                )
        else:
            if cell.typed_unavailable_subclass:
                raise CorpusManifestError(
                    "corpus_manifest_subclass_on_non_typed_unavailable_cell",
                    f"cell {cell.cell_name!r} ({cell.cohort}) "
                    f"is {cell.predictor_bucket} but carries "
                    f"a typed_unavailable_subclass "
                    f"({cell.typed_unavailable_subclass!r}); "
                    f"only predictor_typed_unavailable cells "
                    f"may carry a subclass",
                )
        if (
            cell.measured_cache_initial_state
            not in CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES
        ):
            raise CorpusManifestError(
                "corpus_manifest_measured_cache_initial_state_invalid",
                f"cell {cell.cell_name!r} ({cell.cohort}) has "
                f"measured_cache_initial_state="
                f"{cell.measured_cache_initial_state!r}; expected "
                f"one of "
                f"{sorted(CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES)} "
                f"(empty = use protocol-derived default)",
            )
        prior_cohort = seen_names.get(cell.cell_name)
        if prior_cohort is not None:
            raise CorpusManifestError(
                "corpus_manifest_duplicate_cell_name",
                f"cell_name {cell.cell_name!r} appears in "
                f"both {prior_cohort} and {cell.cohort}",
            )
        seen_names[cell.cell_name] = cell.cohort


def _required_table(
    data: dict[str, Any], dotted_path: str,
) -> dict[str, Any]:
    keys = dotted_path.split(".")
    cursor: Any = data
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            raise CorpusManifestError(
                "corpus_manifest_missing_field",
                f"required TOML table {dotted_path!r} missing",
            )
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"{dotted_path!r} must be a TOML table; got "
            f"{type(cursor).__name__}",
        )
    return cursor


def _required_str(table: dict[str, Any], dotted_path: str) -> str:
    leaf = dotted_path.rsplit(".", 1)[-1]
    if leaf not in table:
        raise CorpusManifestError(
            "corpus_manifest_missing_field",
            f"required field {dotted_path!r} missing",
        )
    value = table[leaf]
    if not isinstance(value, str):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"{dotted_path!r} must be a string; got "
            f"{type(value).__name__}",
        )
    return value


def _optional_str(table: dict[str, Any], key: str) -> str:
    """Return the string value for `key` when present and
    a string; return the empty string when the key is
    absent. Raises `corpus_manifest_schema_violation` when
    the key is present but its value is not a string —
    the loader cannot silently coerce a non-string field
    into the empty default.
    """
    if key not in table:
        return ""
    value = table[key]
    if not isinstance(value, str):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"optional field {key!r} must be a string when "
            f"present; got {type(value).__name__}",
        )
    return value


def _required_str_list(
    table: dict[str, Any], dotted_path: str,
) -> list[str]:
    leaf = dotted_path.rsplit(".", 1)[-1]
    if leaf not in table:
        raise CorpusManifestError(
            "corpus_manifest_missing_field",
            f"required field {dotted_path!r} missing",
        )
    value = table[leaf]
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise CorpusManifestError(
            "corpus_manifest_schema_violation",
            f"{dotted_path!r} must be an array of strings; "
            f"got {type(value).__name__}",
        )
    return list(value)


__all__ = [
    "CALIBRATION_CORPUS_SCHEMA_VERSION",
    "CORPUS_MANIFEST_MEASURED_CACHE_INITIAL_STATES",
    "CORPUS_MANIFEST_PREDICTOR_BUCKETS",
    "CORPUS_MANIFEST_TYPED_REASONS",
    "CalibrationCorpusCell",
    "CalibrationCorpusManifest",
    "CalibrationCorpusMeta",
    "CorpusManifestError",
    "load_calibration_corpus_manifest",
    "validate_calibration_corpus_manifest",
]
