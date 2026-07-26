"""Repeatability aggregator for local-label sidecars.

Groups per-launch label sidecars by cell ``(device, slug, shape_point)``,
checks that each cell has N >= 5 distinct repeat indices with consistent
``binary_sha256``, computes the per-target across-repeat standard
deviation, and emits three indices in the cell-relative output
directory:

- ``label_repeatability.json``: per-cell summary of N, stddev, verdict.
- ``label_admitted.json``: list of admitted cells.
- ``label_excluded.json``: list of excluded cells with typed reasons.

Typed exclusion codes follow the contract:

- ``label_repeatability_missing``: cell has fewer than the required
  number of repeats.
- ``label_repeatability_failed``: per-target standard deviation
  exceeds the preregistered threshold.
- ``label_provenance_inconsistent``: imported sidecars disagree on
  ``binary_sha256`` within the cell.

CLI: ``python tools/local_labels/repeatability_aggregator.py
--root data/local_labels``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

# Sibling import. The aggregator and importer ship as two scripts in the
# same directory, mirroring the convention used elsewhere in tools/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import import_label  # noqa: E402

# Repository-rooted path to the strict admitted-cell schema. The
# aggregator validates every emitted record against the per-record
# sub-schema and aborts emission fail-closed on a validation failure;
# the schema is the producer/consumer wire pin the comparison runner
# relies on.
_REPO_ROOT = _HERE.parents[1]
_LABEL_ADMITTED_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "local_label_admitted.schema.json"
)


class _SchemaUnavailableError(Exception):
    """Fail-closed signal: the strict admitted-cell schema could not be
    loaded. The aggregator catches this and demotes every would-be
    admitted cell to ``label_schema_admission_failed`` rather than
    silently bypassing the strict-schema gate."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def _resolve_schema_path() -> Path:
    """Return the path to the strict admitted-cell schema. The
    ``LABEL_AGGREGATOR_SCHEMA_PATH`` environment variable overrides
    the default repo-rooted location; tests use the override to stage
    missing / malformed / incomplete schema fixtures hermetically."""
    override = os.environ.get("LABEL_AGGREGATOR_SCHEMA_PATH")
    if override:
        return Path(override)
    return _LABEL_ADMITTED_SCHEMA_PATH


def _load_admitted_record_schema(path: Path | None = None) -> dict:
    """Return the per-record admitted-cell sub-schema. Raises
    ``_SchemaUnavailableError`` (fail-closed) when the schema file is
    absent, the JSON does not parse, or the loaded schema is missing
    the ``$defs.admitted_cell`` clause. The aggregator catches the
    exception and demotes every would-be-admitted cell to
    ``label_schema_admission_failed`` — the strict-schema gate must
    not be silently bypassed."""
    if path is None:
        path = _resolve_schema_path()
    if not path.is_file():
        raise _SchemaUnavailableError(
            "missing_file",
            f"admitted-cell schema file not found at {path!s}",
        )
    try:
        with path.open(encoding="utf-8") as fh:
            schema = json.load(fh)
    except json.JSONDecodeError as exc:
        raise _SchemaUnavailableError(
            "malformed_json",
            f"admitted-cell schema JSON did not parse at {path!s}: {exc}",
        ) from exc
    except OSError as exc:
        raise _SchemaUnavailableError(
            "io_error",
            f"could not read admitted-cell schema at {path!s}: {exc}",
        ) from exc
    if not isinstance(schema, dict):
        raise _SchemaUnavailableError(
            "not_an_object",
            f"admitted-cell schema at {path!s} is not a JSON object",
        )
    defs = schema.get("$defs")
    if not isinstance(defs, dict) or "admitted_cell" not in defs:
        raise _SchemaUnavailableError(
            "missing_admitted_cell_def",
            (
                f"admitted-cell schema at {path!s} is missing the "
                f"required `$defs.admitted_cell` definition"
            ),
        )
    return defs["admitted_cell"]


# Per-target standard-deviation threshold in fraction units. Three
# percentage points (0.03) is the preregistered starting value; the
# user may tighten via the open threshold-decision review.
#
# The aggregator uses Python's ``statistics.stdev`` (sample standard
# deviation, Bessel-corrected). The contract treats N >= 5 repeats as
# a sample drawn from the cell's repeated-measurement distribution, so
# the unbiased sample stddev is the correct estimator. The admission
# rule is ``stddev <= 0.03``: a cell whose stddev is exactly the
# threshold remains admissible.
DEFAULT_STDDEV_THRESHOLD: float = 0.03

# Minimum number of distinct repeats per cell. Cells with fewer
# repeats are excluded with typed reason ``label_repeatability_missing``.
DEFAULT_REPEATS_REQUIRED: int = 5

# Pinned protocol version for profiler_version normalization. Mirrors
# the ``ncu_metric_version`` field in
# workloads/simbench_bw/comparison_contract.toml. Sidecars whose raw
# ``profiler_version`` string does not match the
# ``Version 2026.1.<patch>.<build> ...`` family produce a
# ``label_profiler_version_drift`` typed exclusion.
PINNED_NCU_METRIC_VERSION: str = "ncu_2026.1.x"

# Regex matching the raw NCU 2026.1.x version banner. The contract
# admits any 2026.1.<patch>.<build> form; patch / build / suffix
# variants are folded to the pinned protocol form.
_NCU_2026_1_VERSION_RE = re.compile(r"^Version\s+2026\.1\.\d+(?:\.\d+)?\b")

# Required fields on a label_run_manifest_<repeat>_<launch>.json. The
# fail-closed gate rejects manifests with any field missing or null.
_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "source_commit",
    "binary_sha256",
    "kernel_signature_sha256",
    "arch",
    "slug",
    "shape_point",
    "repeat_index",
    "launch_index",
)

# Cell-wide manifest fields whose values must agree across every
# manifest in a single cell directory. A disagreement on any of these
# produces a ``label_provenance_duplicate`` typed exclusion.
_MANIFEST_CELL_INVARIANT_FIELDS: tuple[str, ...] = (
    "source_commit",
    "binary_sha256",
    "kernel_signature_sha256",
)


# The cell key. A BASE cell keeps the historical 3-tuple
# ``(device, slug, shape_point)`` verbatim — byte-identical key arity, so
# every existing consumer that looks up a base cell by a 3-tuple is
# unaffected. A register-cap BUILD VARIANT widens to the 4-tuple
# ``(device, slug, shape_point, variant_id)`` so two build variants of the
# same shape do not collapse into one cell (which would otherwise trip
# ``label_provenance_duplicate`` on their differing binary / kernel
# signatures). The variant component is appended ONLY when present; existing
# call sites read only ``cell_key[0:3]`` and guard ``len(cell_key) > 3``
# before reading the optional variant id.
CellKey = tuple[str, ...]


def _cell_key(arch: str, slug: str, shape_point: str, variant_id: str) -> CellKey:
    """The cell key: the historical 3-tuple for a base cell (``variant_id``
    empty) or the 4-tuple ``(arch, slug, shape_point, variant_id)`` for a
    build variant. Keeping base cells at 3 elements leaves every base-cell
    consumer byte-identical."""
    base = (arch, slug, shape_point)
    return base + (variant_id,) if variant_id else base


# The canonical build-variant identity normalizer lives in its own
# dependency-free module (``variant_identity``) so the lightweight
# ``sidecar_emitter`` — invoked by ``label_driver.sh`` under a bare system
# ``python3`` that may not have jsonschema installed — can share the SAME
# implementation without importing this jsonschema-bearing module. Re-exported
# under the historical private names so the aggregator's helpers and the
# downstream importers (e.g. check_admission_floor) are unchanged.
from variant_identity import (  # noqa: E402
    VARIANT_PREFIX as _VARIANT_PREFIX,
    variant_identity as _variant_identity,
)


def _variant_id_of(record: dict) -> str:
    """The build-variant identity component used for cell GROUPING (the
    canonical ``variant_id``; empty for a base cell). An inconsistent record
    still yields its observed variant_id so the cell exists and
    ``_validate_provenance`` can fail it closed via
    ``_variant_consistency_error``."""
    return _variant_identity(record)[0]


def _variant_consistency_error(record: dict) -> str | None:
    """Non-None reason when a record's ``register_cap``/``variant_id`` disagree
    (or ``variant_id`` is malformed); ``None`` when consistent (or base)."""
    return _variant_identity(record)[2]


@dataclass
class RepeatObservation:
    """One repeat-level observation. When a repeat carries multiple
    launches, the per-launch values are averaged so the stddev that
    feeds the gate measures across-repeat variance, not across-launch
    variance."""

    repeat_index: int
    binary_sha256: str
    cache_hit_rate_l2_sector: float
    cache_hit_rate_l1tex_sector: float | None
    n_launches: int


@dataclass
class CellSummary:
    """Aggregated per-cell view that drives the admission verdict.

    The L2 admission floor is total L2-valid ``n_repeats``; the L1
    admission floor is independent and tracked by ``n_l1tex_repeats``
    (the number of repeats whose per-launch group contains at least
    one ``cache_hit_rate_l1tex_sector`` value). The L1 mean and stddev
    are populated only when ``n_l1tex_repeats >= repeats_required``;
    otherwise both are ``None`` and the aggregator's downstream
    ``_admitted_cell_record`` helper emits the v4 record's
    ``cache_hit_rate_l1tex_sector_pct=null`` +
    ``l1tex_label_status="mean_absent_label_missing"`` shape. The
    schema-side half-pair rejection (L1 metric without CI or vice
    versa) is enforced upstream at the importer layer; this class
    sees only L1-valid or L1-absent repeats by the time it runs.
    """

    cell_key: CellKey
    n_sidecars: int
    n_repeats: int
    n_l1tex_repeats: int
    binary_sha256s: list[str]
    binary_sha256_consistent: bool
    mean_l2_sector: float | None
    stddev_l2_sector: float | None
    mean_l1tex_sector: float | None
    stddev_l1tex_sector: float | None
    repeats: list[RepeatObservation] = field(default_factory=list)
    admission: dict = field(default_factory=dict)
    # Aggregated, per-unit-gated unit-tagged metrics (count/ns/cycles and any
    # non-flat-field pct family) admitted onto this cell. Empty for legacy
    # cells whose sidecars carry only the flat L1/L2 fraction fields, so the
    # admitted record then OMITS the optional unit_tagged_metrics block and
    # stays byte-compatible with the pre-v5 shape apart from schema_version.
    unit_tagged_metrics: list[dict] = field(default_factory=list)


def _collect_sidecars(root: Path) -> list[Path]:
    """Recursively gather every per-launch sidecar under ``root`` whose
    filename matches the contract layout ``label_<repeat>_<launch>.json``.

    The match deliberately excludes ``label_run_manifest_*_*.json``
    files, which share the ``label_*_*.json`` glob root but live in a
    separate file family (provenance manifest vs. measured sidecar).

    Sweep-subtree exclusion: per-launch sidecars under
    ``<root>/sweeps/`` belong to already-selected per-config
    cohorts (aggregator outputs under
    ``sweeps/<sweep>/cell=*/cfg=*/_aggregator_out/``;
    committed admitted-combined / sweep_round11_ac4 trees;
    operator-local raw NCU sidecars under gitignored
    ``cell=*/cfg=*/`` paths). Mixing those per-config cohorts
    into the root corpus aggregation groups different
    (warmup, cache_control) configs under the same
    (device, slug, shape_point) key and corrupts the resulting
    per-cell repeatability verdict. Operators who explicitly
    target a sweep subdir with ``--root <sweep_subdir>``
    still get the per-config aggregation; only the
    default-root walk excludes ``<root>/sweeps/``.
    """
    if not root.exists():
        return []
    # Restrict to the contract-pinned per-launch label filename
    # `label_<repeat>_<launch>.json` where both indices are
    # non-negative integers. The bare `label_*_*.json` glob also
    # matched (a) committed sweep aggregator outputs under
    # `data/local_labels/sweeps/.../config=*/_aggregator_out/`
    # which carry filenames like `label_admitted.json` /
    # `label_excluded.json` / `label_repeatability.json`, AND
    # (b) any future round-introduced `label_<word>_<word>.json`
    # convention. Numeric-only validation keeps the per-launch
    # cohort intact.
    import re
    NUMERIC_LABEL_RE = re.compile(r"^label_\d+_\d+\.json$")
    sweeps_subtree = root / "sweeps"
    return sorted(
        path for path in root.rglob("label_*_*.json")
        if NUMERIC_LABEL_RE.match(path.name)
        and sweeps_subtree not in path.parents
    )


def _collect_manifests(root: Path) -> dict[CellKey, list[tuple[Path, dict]]]:
    """Gather every ``label_run_manifest_<repeat>_<launch>.json`` under
    ``root`` and group by cell key. Manifests that fail to load as
    valid JSON or lack a required field are excluded; cells whose
    manifest set is empty after this filter trigger
    ``label_provenance_unavailable`` downstream.

    Sweep-subtree exclusion (parallel to _collect_sidecars): when
    --root is the default `data/local_labels/`, skip the
    `<root>/sweeps/` subtree so per-config sweep cohorts don't
    mix into the root corpus aggregation.
    """
    if not root.exists():
        return {}
    out: dict[CellKey, list[tuple[Path, dict]]] = defaultdict(list)
    sweeps_subtree = root / "sweeps"
    for path in sorted(root.rglob("label_run_manifest_*_*.json")):
        if sweeps_subtree in path.parents:
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(manifest, dict):
            continue
        if any(
            field not in manifest or manifest.get(field) in (None, "")
            for field in _MANIFEST_REQUIRED_FIELDS
        ):
            continue
        cell = _cell_key(
            str(manifest["arch"]),
            str(manifest["slug"]),
            str(manifest["shape_point"]),
            _variant_id_of(manifest),
        )
        out[cell].append((path, manifest))
    return dict(out)


def _normalize_profiler_version(raw: str) -> str | None:
    """Return the contract's pinned ``ncu_metric_version`` form when
    the raw NCU version banner matches the
    ``Version 2026.1.<patch>(.<build>)`` family. Return ``None`` to
    signal a profiler-version drift (the caller emits
    ``label_profiler_version_drift``)."""
    if not isinstance(raw, str):
        return None
    if _NCU_2026_1_VERSION_RE.match(raw):
        return PINNED_NCU_METRIC_VERSION
    return None


def _validate_provenance(
    cell_key: CellKey,
    sidecars: list[dict],
    manifests: list[tuple[Path, dict]],
) -> dict | None:
    """Return an admission-exclusion dict on a provenance failure,
    or ``None`` when the cell's manifests are well-formed and agree
    with the per-launch sidecars.

    The validator runs five checks in fail-fast order:

    1. ``label_provenance_unavailable``: the cell has zero usable
       manifests.
    2. ``label_provenance_duplicate``: two or more manifests in the
       cell directory disagree on a cell-invariant field.
    3. ``label_provenance_per_launch_missing``: a per-launch sidecar
       has no matching ``label_run_manifest_<repeat>_<launch>.json``;
       the cell-invariant fields cannot bind a sidecar that lacks
       its own per-launch manifest.
    4. ``label_provenance_mismatch``: a per-launch sidecar's
       ``binary_sha256`` disagrees with the manifest covering the
       same (repeat, launch) tuple.
    5. ``label_profiler_version_drift``: a per-launch sidecar's raw
       ``profiler_version`` does not match the contract's pinned
       protocol form.
    """
    if not manifests:
        return {
            "admitted": False,
            "typed_reason": "label_provenance_unavailable",
            "detail": (
                f"cell {cell_key} has no label_run_manifest_*_*.json "
                f"files (or every candidate manifest is missing a "
                f"required field)"
            ),
        }
    # Build-variant identity consistency: a record whose register_cap and
    # variant_id disagree (or whose variant_id is malformed) must fail the
    # cell CLOSED before any admitted record is emitted -- otherwise the
    # grouping (keyed on variant_id) and the admitted register_cap could
    # describe two different build variants for one cell.
    for _path, manifest in manifests:
        variant_error = _variant_consistency_error(manifest)
        if variant_error is not None:
            return {
                "admitted": False,
                "typed_reason": "label_variant_identity_inconsistent",
                "detail": (
                    f"cell {cell_key} run-manifest has inconsistent "
                    f"build-variant identity: {variant_error}"
                ),
            }
    for sidecar in sidecars:
        variant_error = _variant_consistency_error(sidecar)
        if variant_error is not None:
            return {
                "admitted": False,
                "typed_reason": "label_variant_identity_inconsistent",
                "detail": (
                    f"cell {cell_key} sidecar has inconsistent "
                    f"build-variant identity: {variant_error}"
                ),
            }
    for field in _MANIFEST_CELL_INVARIANT_FIELDS:
        values = sorted({str(m.get(field, "")) for _path, m in manifests})
        if len(values) > 1:
            return {
                "admitted": False,
                "typed_reason": "label_provenance_duplicate",
                "detail": (
                    f"cell {cell_key} has manifests disagreeing on "
                    f"{field}: {values}"
                ),
            }
    manifest_by_tuple: dict[tuple[int, int], dict] = {}
    for _path, manifest in manifests:
        try:
            key = (int(manifest["repeat_index"]), int(manifest["launch_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        manifest_by_tuple[key] = manifest
    for sidecar in sidecars:
        try:
            sidecar_tuple = (
                int(sidecar.get("repeat_index")),
                int(sidecar.get("launch_index")),
            )
        except (TypeError, ValueError):
            return {
                "admitted": False,
                "typed_reason": "label_provenance_per_launch_missing",
                "detail": (
                    f"cell {cell_key} per-launch sidecar lacks usable "
                    f"(repeat_index, launch_index) fields; cannot bind "
                    f"to a per-launch manifest"
                ),
            }
        if sidecar_tuple not in manifest_by_tuple:
            return {
                "admitted": False,
                "typed_reason": "label_provenance_per_launch_missing",
                "detail": (
                    f"cell {cell_key} per-launch sidecar at "
                    f"(repeat_index, launch_index)={sidecar_tuple!r} has "
                    f"no matching label_run_manifest_*_*.json; the "
                    f"cell-invariant manifest fields cannot bind a "
                    f"sidecar without its own per-launch manifest"
                ),
            }
        per_launch_manifest = manifest_by_tuple[sidecar_tuple]
        sidecar_binary = sidecar.get("binary_sha256", "")
        if sidecar_binary != per_launch_manifest.get("binary_sha256", ""):
            return {
                "admitted": False,
                "typed_reason": "label_provenance_mismatch",
                "detail": (
                    f"cell {cell_key} per-launch sidecar at "
                    f"(repeat_index, launch_index)={sidecar_tuple!r} has "
                    f"binary_sha256={sidecar_binary!r} but its matching "
                    f"manifest has binary_sha256="
                    f"{per_launch_manifest.get('binary_sha256', '')!r}"
                ),
            }
    for sidecar in sidecars:
        raw_version = sidecar.get("profiler_version", "")
        if _normalize_profiler_version(raw_version) is None:
            return {
                "admitted": False,
                "typed_reason": "label_profiler_version_drift",
                "detail": (
                    f"cell {cell_key} per-launch sidecar "
                    f"profiler_version={raw_version!r} does not match "
                    f"the pinned protocol form "
                    f"{PINNED_NCU_METRIC_VERSION!r}"
                ),
            }
    return None


def _admitted_evidence_class(sidecars: list[dict]) -> str | None:
    """Return one explicit class only when every sidecar agrees on it."""
    classes = [sidecar.get("evidence_class") for sidecar in sidecars]
    if not classes or any(not isinstance(value, str) or not value for value in classes):
        return None
    if len(set(classes)) != 1:
        return None
    return classes[0]


def _admitted_cell_record(
    summary: "CellSummary",
    manifests: list[tuple[Path, dict]],
    sidecars: list[dict],
) -> dict:
    """Compose the admitted-cell dict for ``label_admitted.json``.
    Carries the runner-joinable 7-field key plus the percent-form L2
    hit rate, the per-layer L1 mean + status pair, the n_repeats /
    stddev pair, and the schema_version pin."""
    canonical = manifests[0][1]
    raw_versions = [sidecars[0].get("profiler_version", "")] if sidecars else [""]
    normalized = _normalize_profiler_version(raw_versions[0]) or PINNED_NCU_METRIC_VERSION
    cache_hit_rate_l2_pct = (
        round(float(summary.mean_l2_sector) * 100.0, 6)
        if summary.mean_l2_sector is not None
        else 0.0
    )
    # Per-layer L1 emission. The schema's `l1tex_label_status`
    # enum admits four values:
    #   - mean_present: at least one per-repeat sidecar carried
    #     a non-zero cache_hit_rate_l1tex_sector value (cohort
    #     mean defined + nonzero).
    #   - all_zero: every per-repeat sidecar carried a 0.0
    #     cache_hit_rate_l1tex_sector value (e.g. vendor_sgemm_cublas
    #     whose cublas-backed kernel bypasses the L1tex cache
    #     pipeline). The mean is 0.0 and the downstream
    #     comparison runner consumes the cell normally for L2
    #     prediction; the runner's L1 join uses the existing
    #     epsilon denominator against the 0.0 ground truth.
    #   - mean_absent_label_missing: sidecars exist (so L2 mean
    #     computed cleanly) but no repeat carried an L1 value
    #     at all.
    #   - mean_absent_provenance_missing: reserved for the
    #     migration path where per-repeat provenance is lost
    #     entirely; the live aggregator does not emit this.
    if summary.mean_l1tex_sector is not None:
        cache_hit_rate_l1tex_pct: float | None = round(
            float(summary.mean_l1tex_sector) * 100.0, 6,
        )
        # Distinguish all_zero from mean_present: when every
        # per-repeat L1 value was captured but each was exactly
        # 0.0, the resulting mean is 0.0 and the stddev is also
        # 0.0. The runner.py-side downstream code relies on the
        # all_zero status to explain the cell correctly (per the
        # schema description).
        if (
            cache_hit_rate_l1tex_pct == 0.0
            and summary.stddev_l1tex_sector is not None
            and summary.stddev_l1tex_sector == 0.0
        ):
            l1tex_label_status = "all_zero"
        else:
            l1tex_label_status = "mean_present"
    else:
        cache_hit_rate_l1tex_pct = None
        l1tex_label_status = "mean_absent_label_missing"
    evidence_class = _admitted_evidence_class(sidecars)
    record = {
        "schema_version": 5,
        "slug": summary.cell_key[1],
        "shape_point": summary.cell_key[2],
        "binary_sha256": canonical["binary_sha256"],
        "source_commit": canonical["source_commit"],
        "kernel_signature_sha256": canonical["kernel_signature_sha256"],
        "arch": summary.cell_key[0],
        "ncu_metric_version": PINNED_NCU_METRIC_VERSION,
        "profiler_version": normalized,
        "cache_hit_rate_l2_sector_pct": cache_hit_rate_l2_pct,
        "cache_hit_rate_l1tex_sector_pct": cache_hit_rate_l1tex_pct,
        "l1tex_label_status": l1tex_label_status,
        "n_repeats": summary.n_repeats,
        "stddev_l2_sector": summary.stddev_l2_sector,
        "stddev_l1tex_sector": summary.stddev_l1tex_sector,
        "evidence_kind": "live" if evidence_class == "campaign" else "synthetic",
    }
    if evidence_class is not None:
        record["evidence_class"] = evidence_class
    # unit-tagged metrics that survived their per-unit repeatability gate.
    # Legacy L1/L2-only cells omit the field, keeping the record otherwise
    # byte-identical to its pre-v5 shape apart from schema_version.
    if summary.unit_tagged_metrics:
        record["unit_tagged_metrics"] = summary.unit_tagged_metrics
    # Optional build-variant identity (additive). Present ONLY for a
    # register-cap build-variant cell; a base cell's record is byte-identical
    # to before. Both fields come from the SINGLE normalizer (the cell has
    # already passed the provenance gate, so the pair is consistent), so the
    # admitted variant_id and register_cap can never disagree.
    variant_id, register_cap, _error = _variant_identity(canonical)
    if variant_id:
        record["variant_id"] = variant_id
        if register_cap is not None:
            record["register_cap"] = register_cap
    return record


def _load_sidecars_by_cell(
    paths: list[Path],
) -> tuple[dict[CellKey, list[dict]], list[tuple[Path, import_label.ValidationResult]]]:
    """Validate each sidecar and group the valid ones by cell key.

    Invalid sidecars are returned separately so the caller can record
    them under ``rejected_sidecars`` in the per-cell summary index.
    """
    by_cell: dict[CellKey, list[dict]] = defaultdict(list)
    rejected: list[tuple[Path, import_label.ValidationResult]] = []
    for path in paths:
        result = import_label.validate_path(path)
        if not result.ok:
            rejected.append((path, result))
            continue
        with path.open("r", encoding="utf-8") as fh:
            sidecar = json.load(fh)
        cell = _cell_key(
            str(sidecar["device"]),
            str(sidecar["slug"]),
            str(sidecar["shape_point"]),
            _variant_id_of(sidecar),
        )
        by_cell[cell].append(sidecar)
    return dict(by_cell), rejected


def _dispersion_for_rule(
    rule: str, values: list[float], mean: float
) -> float | None:
    """Across-repeat dispersion under a per-unit repeatability rule.

    - ``stddev_abs``: sample standard deviation (fraction units), mirroring the
      L2/L1 hit-rate gate.
    - ``coefficient_of_variation``: stddev / |mean| (for cycles / ns). When the
      mean is exactly 0, returns 0.0 iff every value is identical (else None =
      undefined = gate-fail).
    - ``relative_tolerance``: (max - min) / |mean| (for counts). Returns 0.0
      when every value is identical (covers all-zero counts), else None when
      the mean is 0 with spread.

    Returns ``None`` when the dispersion is undefined (fewer than two values,
    or a 0 mean with spread); the caller treats ``None`` as a gate failure.
    """
    if len(values) < 2:
        return None
    if rule == "stddev_abs":
        return statistics.stdev(values)
    if rule == "coefficient_of_variation":
        sd = statistics.stdev(values)
        if mean == 0:
            return 0.0 if sd == 0 else None
        return sd / abs(mean)
    if rule == "relative_tolerance":
        lo, hi = min(values), max(values)
        if hi == lo:
            return 0.0
        if mean == 0:
            return None
        return (hi - lo) / abs(mean)
    return None


def _aggregate_unit_tagged_metrics(
    by_repeat: dict[int, list[dict]],
    repeats_required: int,
) -> list[dict]:
    """Aggregate the optional per-launch ``unit_tagged_metrics`` blocks into
    per-cell admitted records, gated by each metric's per-unit repeatability
    rule resolved from the committed metric registry.

    Within a repeat, a metric's per-launch values are averaged (matching the
    L1/L2 across-repeat-variance discipline); across repeats the mean and a
    rule-specific dispersion are computed. A metric is admitted onto the cell
    only when it appears in at least ``repeats_required`` repeats AND its
    dispersion is within the registry threshold for its rule. Metrics that
    fall short are silently dropped from the cell (they do not regress the
    independent L2 admission). Unregistered metrics are skipped (the importer
    rejects them upstream). Returns the sorted list of admitted records.
    """
    from metric_registry import load_registry

    registry = load_registry()
    per_metric_repeat_means: dict[str, dict[int, float]] = defaultdict(dict)
    for repeat_index, group in by_repeat.items():
        repeat_vals: dict[str, list[float]] = defaultdict(list)
        for sidecar in group:
            for rec in sidecar.get("unit_tagged_metrics") or []:
                if not isinstance(rec, dict):
                    continue
                ncu_metric = rec.get("ncu_metric")
                value = rec.get("value")
                if ncu_metric is None or value is None:
                    continue
                repeat_vals[ncu_metric].append(float(value))
        for ncu_metric, vals in repeat_vals.items():
            per_metric_repeat_means[ncu_metric][repeat_index] = statistics.fmean(
                vals
            )

    admitted: list[dict] = []
    for ncu_metric in sorted(per_metric_repeat_means):
        entry = registry.get_by_ncu_metric(ncu_metric)
        if entry is None:
            continue
        repeat_means = per_metric_repeat_means[ncu_metric]
        values = [repeat_means[r] for r in sorted(repeat_means)]
        if len(values) < repeats_required:
            continue
        mean = statistics.fmean(values)
        rule = entry.repeatability_rule
        threshold = registry.threshold_for(entry)
        dispersion = _dispersion_for_rule(rule, values, mean)
        if dispersion is None or dispersion > threshold:
            continue
        admitted.append(
            {
                "ncu_metric": ncu_metric,
                "unit": entry.unit,
                "value": round(mean, 6),
                "dispersion": round(dispersion, 6),
                "repeatability_rule": rule,
                "n_repeats": len(values),
            }
        )
    return admitted


def _summarize_cell(
    cell_key: CellKey,
    sidecars: list[dict],
    *,
    repeats_required: int,
) -> CellSummary:
    """Compute per-cell statistics across repeats for the L2 (mandatory)
    and L1/TEX (optional) targets.

    L1 admission gate: the L1 mean + stddev are populated ONLY when
    ``n_l1tex_repeats >= repeats_required`` (the same floor that
    drives L2 admission). With fewer L1-bearing repeats, the L1 mean
    and stddev are both ``None`` and the downstream
    ``_admitted_cell_record`` helper emits the v4 record's null L1
    mean + ``l1tex_label_status="mean_absent_label_missing"`` shape.
    Partial L1 evidence (e.g., 1-of-5 L1-bearing repeats) must NOT
    promote to ``mean_present`` because (a) the stddev is undefined
    for n<2 and degenerate for n<repeats_required, and (b) an
    under-evidenced L1 mean would falsely contribute to the L1
    relative error in downstream comparison/cohort artifacts.

    L2 admission is independent: it stays driven by the total L2-valid
    ``n_repeats``, so absent or partial L1 evidence does NOT regress
    L2 coverage.
    """
    binary_sha256s = sorted({s["binary_sha256"] for s in sidecars})
    by_repeat: dict[int, list[dict]] = defaultdict(list)
    for sidecar in sidecars:
        by_repeat[int(sidecar["repeat_index"])].append(sidecar)

    repeats: list[RepeatObservation] = []
    for repeat_index in sorted(by_repeat):
        group = by_repeat[repeat_index]
        l2_values = [float(s["cache_hit_rate_l2_sector"]) for s in group]
        l1_values = [
            float(s["cache_hit_rate_l1tex_sector"])
            for s in group
            if "cache_hit_rate_l1tex_sector" in s
        ]
        repeats.append(
            RepeatObservation(
                repeat_index=repeat_index,
                binary_sha256=group[0]["binary_sha256"],
                cache_hit_rate_l2_sector=statistics.fmean(l2_values),
                cache_hit_rate_l1tex_sector=(
                    statistics.fmean(l1_values) if l1_values else None
                ),
                n_launches=len(group),
            )
        )

    n_repeats = len(repeats)
    n_l1tex_repeats = sum(
        1 for r in repeats
        if r.cache_hit_rate_l1tex_sector is not None
    )
    if n_repeats >= 2:
        l2_per_repeat = [r.cache_hit_rate_l2_sector for r in repeats]
        stddev_l2: float | None = statistics.stdev(l2_per_repeat)
        mean_l2: float | None = statistics.fmean(l2_per_repeat)
    else:
        stddev_l2 = None
        mean_l2 = (
            statistics.fmean([r.cache_hit_rate_l2_sector for r in repeats])
            if repeats
            else None
        )
    # L1 admission gate: require the configured repeat floor of
    # L1-bearing repeats before computing the L1 mean + stddev.
    # Below the floor, both are None and the v4 admitted record
    # carries `l1tex_label_status="mean_absent_label_missing"`.
    # Single-repeat guard mirrors the L2 path at line 542: stddev
    # is undefined for n<2 (statistics.stdev raises StatisticsError).
    # When the configured `--repeats-required` is 1, the mean is
    # still well-defined but the stddev is None.
    if n_l1tex_repeats >= repeats_required:
        l1_values = [
            r.cache_hit_rate_l1tex_sector
            for r in repeats
            if r.cache_hit_rate_l1tex_sector is not None
        ]
        if len(l1_values) >= 2:
            stddev_l1: float | None = statistics.stdev(l1_values)
        else:
            stddev_l1 = None
        mean_l1: float | None = (
            statistics.fmean(l1_values) if l1_values else None
        )
    else:
        stddev_l1 = None
        mean_l1 = None

    return CellSummary(
        cell_key=cell_key,
        n_sidecars=len(sidecars),
        n_repeats=n_repeats,
        n_l1tex_repeats=n_l1tex_repeats,
        binary_sha256s=binary_sha256s,
        binary_sha256_consistent=len(binary_sha256s) == 1,
        mean_l2_sector=mean_l2,
        stddev_l2_sector=stddev_l2,
        mean_l1tex_sector=mean_l1,
        stddev_l1tex_sector=stddev_l1,
        repeats=repeats,
        unit_tagged_metrics=_aggregate_unit_tagged_metrics(
            by_repeat, repeats_required
        ),
    )


def _admit(
    summary: CellSummary,
    repeats_required: int,
    stddev_threshold: float,
) -> dict:
    """Compute the admission verdict for a cell. Provenance check fires
    first because a multi-binary cell is malformed regardless of the
    number of repeats; missing-N fires before stddev because the
    stddev is undefined for a single-repeat cell."""
    if not summary.binary_sha256_consistent:
        return {
            "admitted": False,
            "typed_reason": "label_provenance_inconsistent",
            "detail": (
                f"cell {summary.cell_key} has "
                f"{len(summary.binary_sha256s)} distinct binary_sha256 "
                f"values across imported sidecars: "
                f"{summary.binary_sha256s}"
            ),
        }

    if summary.n_repeats < repeats_required:
        return {
            "admitted": False,
            "typed_reason": "label_repeatability_missing",
            "detail": (
                f"cell {summary.cell_key} has {summary.n_repeats} "
                f"repeats; contract requires N >= {repeats_required}."
            ),
        }

    l2 = summary.stddev_l2_sector
    if l2 is None or l2 > stddev_threshold:
        return {
            "admitted": False,
            "typed_reason": "label_repeatability_failed",
            "detail": (
                f"cell {summary.cell_key} L2 stddev "
                f"{l2 if l2 is not None else 'unavailable'} exceeds "
                f"threshold {stddev_threshold}."
            ),
        }

    l1 = summary.stddev_l1tex_sector
    if l1 is not None and l1 > stddev_threshold:
        return {
            "admitted": False,
            "typed_reason": "label_repeatability_failed",
            "detail": (
                f"cell {summary.cell_key} L1/TEX stddev {l1} exceeds "
                f"threshold {stddev_threshold}."
            ),
        }

    return {
        "admitted": True,
        "typed_reason": "",
        "detail": "",
    }


def aggregate(
    quarantine_root: Path,
    repeats_required: int = DEFAULT_REPEATS_REQUIRED,
    stddev_threshold: float = DEFAULT_STDDEV_THRESHOLD,
) -> tuple[
    dict, dict, dict, list[tuple[Path, import_label.ValidationResult]]
]:
    """Walk ``quarantine_root``, summarize each cell, and return
    ``(repeatability_summary, admitted_index, excluded_index,
    rejected_sidecars)``. The caller writes the three dicts to disk."""
    sidecar_paths = _collect_sidecars(quarantine_root)
    by_cell, rejected = _load_sidecars_by_cell(sidecar_paths)
    manifests_by_cell = _collect_manifests(quarantine_root)

    summaries: list[CellSummary] = []
    cell_manifests: dict[CellKey, list[tuple[Path, dict]]] = {}
    for cell_key in sorted(by_cell):
        summary = _summarize_cell(
            cell_key,
            by_cell[cell_key],
            repeats_required=repeats_required,
        )
        manifests = manifests_by_cell.get(cell_key, [])
        cell_manifests[cell_key] = manifests
        provenance_failure = _validate_provenance(
            cell_key, by_cell[cell_key], manifests,
        )
        if provenance_failure is not None:
            summary.admission = provenance_failure
        else:
            summary.admission = _admit(
                summary, repeats_required, stddev_threshold,
            )
        summaries.append(summary)

    # Schema-fail-closed admission gate (review-fix-7 ordering).
    #
    # The repeatability/admitted/excluded artifacts are emitted by the
    # SAME aggregator invocation and must agree on every cell's
    # verdict. Earlier versions built the repeatability payload BEFORE
    # the schema-fail-closed demotion ran, so a schema-unavailable
    # cell appeared as `admission.admitted: true` in
    # label_repeatability.json while label_admitted.json was empty
    # and label_excluded.json correctly reported
    # label_schema_admission_failed. Run the demotion FIRST, so each
    # cell's `admission` reaches its final state before any artifact
    # captures it.
    admitted_cells: list[dict] = []
    record_schema: dict | None
    schema_unavailable_detail: str | None = None
    try:
        record_schema = _load_admitted_record_schema()
    except _SchemaUnavailableError as exc:
        # Fail-closed: an unavailable / malformed / incomplete schema
        # demotes every would-be-admitted cell to
        # `label_schema_admission_failed`. The strict-schema gate is the
        # producer/consumer wire pin and must not be silently bypassed
        # when the schema file disappears or is corrupted.
        record_schema = None
        schema_unavailable_detail = (
            f"admitted-cell schema unavailable ({exc.reason}): {exc.detail}"
        )
    schema_validation_failures: list[dict] = []
    for s in summaries:
        if not s.admission.get("admitted"):
            continue
        manifests = cell_manifests.get(s.cell_key, [])
        sidecars = by_cell.get(s.cell_key, [])
        if schema_unavailable_detail is not None:
            s.admission = {
                "admitted": False,
                "typed_reason": "label_schema_admission_failed",
                "detail": (
                    f"cell {s.cell_key}: {schema_unavailable_detail}"
                ),
            }
            schema_validation_failures.append({
                "cell_key": s.cell_key,
                "detail": schema_unavailable_detail,
            })
            continue
        record = _admitted_cell_record(s, manifests, sidecars)
        try:
            jsonschema.validate(instance=record, schema=record_schema)
        except jsonschema.ValidationError as exc:
            # Fail-closed: a record that does not validate against
            # the strict admitted-cell schema must not enter the
            # admitted set. Demote the cell to excluded with a
            # typed reason naming the schema-validation failure.
            s.admission = {
                "admitted": False,
                "typed_reason": "label_schema_admission_failed",
                "detail": (
                    f"cell {s.cell_key} record failed strict schema "
                    f"validation: {exc.message}"
                ),
            }
            schema_validation_failures.append({
                "cell_key": s.cell_key,
                "detail": exc.message,
            })
            continue
        admitted_cells.append(record)
    admitted = {
        "schema_version": 5,
        "cells": admitted_cells,
    }

    # Build the repeatability payload AFTER schema demotion finishes
    # so every cell's `admission` reflects the post-demotion verdict.
    repeatability = {
        "schema_version": 1,
        "repeats_required": repeats_required,
        "stddev_threshold": stddev_threshold,
        "cells": [
            {
                "device": s.cell_key[0],
                "slug": s.cell_key[1],
                "shape_point": s.cell_key[2],
                "n_sidecars": s.n_sidecars,
                "n_repeats": s.n_repeats,
                "n_l1tex_repeats": s.n_l1tex_repeats,
                "binary_sha256_consistent": s.binary_sha256_consistent,
                "binary_sha256s": s.binary_sha256s,
                "mean_l2_sector": s.mean_l2_sector,
                "stddev_l2_sector": s.stddev_l2_sector,
                "mean_l1tex_sector": s.mean_l1tex_sector,
                "stddev_l1tex_sector": s.stddev_l1tex_sector,
                "admission": s.admission,
                # Build-variant identity: emitted ONLY for a variant cell so
                # two same-shape variants are distinguishable here; base-cell
                # rows are byte-identical to before.
                **({"variant_id": s.cell_key[3]} if len(s.cell_key) > 3 and s.cell_key[3] else {}),
            }
            for s in summaries
        ],
        "rejected_sidecars": [
            {
                "path": str(path),
                "typed_reason": result.typed_reason,
                "detail": result.detail,
            }
            for path, result in rejected
        ],
    }

    excluded = {
        "schema_version": 1,
        "cells": [
            {
                "device": s.cell_key[0],
                "slug": s.cell_key[1],
                "shape_point": s.cell_key[2],
                "n_repeats": s.n_repeats,
                "typed_reason": s.admission["typed_reason"],
                "detail": s.admission["detail"],
                "stddev_l2_sector": s.stddev_l2_sector,
                "stddev_l1tex_sector": s.stddev_l1tex_sector,
                "binary_sha256s": s.binary_sha256s,
                # Build-variant identity: emitted ONLY for a variant cell;
                # base-cell rows are byte-identical to before.
                **({"variant_id": s.cell_key[3]} if len(s.cell_key) > 3 and s.cell_key[3] else {}),
            }
            for s in summaries
            if not s.admission.get("admitted")
        ],
    }

    return repeatability, admitted, excluded, rejected


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-launch local-label sidecars under a quarantine "
            "root, gate by repeatability, and emit summary + admitted + "
            "excluded indices."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/local_labels"),
        help="Quarantine root (default: data/local_labels).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for the three indices. Defaults to <root>.",
    )
    parser.add_argument(
        "--repeats-required",
        type=int,
        default=DEFAULT_REPEATS_REQUIRED,
        help=f"Minimum N per cell (default: {DEFAULT_REPEATS_REQUIRED}).",
    )
    parser.add_argument(
        "--stddev-threshold",
        type=float,
        default=DEFAULT_STDDEV_THRESHOLD,
        help=(
            "Per-target stddev threshold in fraction units "
            f"(default: {DEFAULT_STDDEV_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--arch",
        default=None,
        help=(
            "Optional architecture string (e.g. `sm_86`, "
            "`sm_120-laptop`). When set, the aggregator validates "
            "the running script against `tools/arch_capabilities.json` "
            "before doing any work; on an unsupported arch the "
            "aggregator emits a typed `arch_unsupported` JSON to "
            "stderr and exits non-zero rather than silently "
            "returning a default verdict."
        ),
    )
    args = parser.parse_args(argv)

    if args.arch is not None:
        # Lazy import keeps the helper out of the import chain when
        # the operator does not request arch validation. The
        # repository-relative path resolves through the helper's
        # manifest so the aggregator does not duplicate the
        # supported_arches list.
        _HELPER_DIR = Path(__file__).resolve().parent.parent
        if str(_HELPER_DIR) not in sys.path:
            sys.path.insert(0, str(_HELPER_DIR))
        import arch_capabilities  # type: ignore[import-not-found]  # noqa: E402

        typed = arch_capabilities.assert_arch_supported(
            path="tools/local_labels/repeatability_aggregator.py",
            arch=args.arch,
        )
        if typed is not None:
            sys.stderr.write(
                json.dumps(typed, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
            return 1

    out_dir = args.out if args.out is not None else args.root
    repeatability, admitted, excluded, _ = aggregate(
        quarantine_root=args.root,
        repeats_required=args.repeats_required,
        stddev_threshold=args.stddev_threshold,
    )
    _write(out_dir / "label_repeatability.json", repeatability)
    _write(out_dir / "label_admitted.json", admitted)
    _write(out_dir / "label_excluded.json", excluded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
