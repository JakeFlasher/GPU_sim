"""Run the active sm_86 predictor/comparison cohort.

The historical sm_120 activation path is retained in source and evidence, but
is blocked by the sm120-template safety gate. Any sm_120-bearing request
returns a typed-unavailable diagnostic before bundle production, calibration,
or comparison scoring; an explicitly sm_86-scoped mixed request continues with
its sm_86 subset. No retained evidence is deleted or rewritten.

Per-cell predictor bundles land at:
  data/local_labels/sweeps/<sweep_id>/predictor_bundles/<cell>.json

The targets-never-features boundary applies here: NCU is NOT invoked; only the
predictor + comparison library calls run. The predictor reads NVBit traces and
emits modeled-hit predictions. Living under ``tools/local_labels/`` keeps this
orchestrator co-located with the other label-driver code; it does NOT perform
hardware collection (no NCU, no nvidia-smi mutation, no privileged CUPTI).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUITE_ROOT = _REPO_ROOT / "workloads" / "simbench_bw"
_RESULTS_ROOT = _SUITE_ROOT / "results"
_HRD_PROFILER_DIR = _REPO_ROOT / "tools" / "hrd_profiler"

sys.path.insert(0, str(_SUITE_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "sector_stream"))
sys.path.insert(0, str(_HRD_PROFILER_DIR))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "local_labels"))

import cache_predictor as cp  # noqa: E402
import comparison_runner as cr  # noqa: E402
import comparison_contract_overlay as cco  # noqa: E402
import device_properties  # noqa: E402
from tools.local_labels.immutable_artifact import (  # noqa: E402
    ImmutableArtifactError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
)
from tools.nvbit_tracer.trace_admission import (  # noqa: E402
    AdmittedTraceSources,
    TraceAdmissionError,
    open_admitted_trace_sources,
    require_trace_admission,
)
from tools.sector_stream.consume_trace import (  # noqa: E402
    AdmittedSectorStream,
    consume_admitted_trace_sources,
)

_HARD_TRACE_SOURCE_ERRORS = (
    TraceAdmissionError,
    ImmutableArtifactLimitError,
    ImmutableArtifactResourceError,
    ImmutableArtifactError,
)

try:
    from tools.local_labels.sm120_template_safety import (  # noqa: E402
        SM120_ARCH,
        SM120_TEMPLATE_UNAVAILABLE_REASON,
        Sm120TemplateUnavailable,
        canonical_arch,
        is_sm120_template_arch,
        unavailable_result,
    )
except ModuleNotFoundError:  # direct script execution from tools/local_labels
    from sm120_template_safety import (  # noqa: E402
        SM120_ARCH,
        SM120_TEMPLATE_UNAVAILABLE_REASON,
        Sm120TemplateUnavailable,
        canonical_arch,
        is_sm120_template_arch,
        unavailable_result,
    )

# The cohort manifest validator + dataclass live under
# `tools/sm_86_predictor_accuracy_recovery_v2/`. Inject that path
# into sys.path here so the in-process import succeeds without
# requiring the runner to be invoked from the repo root.
_COHORT_MANIFEST_TOOLS_DIR = (
    _REPO_ROOT / "tools" / "sm_86_predictor_accuracy_recovery_v2"
)
if str(_COHORT_MANIFEST_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_COHORT_MANIFEST_TOOLS_DIR))

import validate_cohort_manifest as vcm  # noqa: E402
import publication_gate as pg  # noqa: E402
import label_loader as _label_loader_mod  # noqa: E402


from typing import NamedTuple


class PredictorStageRouting(NamedTuple):
    """Routing decision for the driver's predictor stage.

    ``outcome`` is one of:
      * ``"legacy"`` -- the cohort contains only active sm_86 cells; the
        legacy ``predict_one_cell`` loop runs unchanged.
      * ``"sm120_template_unavailable"`` -- at least one sm_120 cell was
        requested. The retained sm_120 template lane is represented as a
        typed-unavailable result and the producer/comparison stages are never
        invoked for it.
    """

    outcome: str
    n_sm120: int = 0
    n_total: int = 0
    diagnostic: str = ""
    detail: str = ""


def _route_predictor_stage(
    *,
    headline_cohort: list[tuple[str, str, str]],
    target_arch: str | None,
) -> PredictorStageRouting:
    """Decide whether the active sm_86 loop may run.

    sm_120 is deliberately a hard safety boundary. A cohort containing any
    spelling of that architecture never routes to the former first-class
    producer path, even when historical bundles or calibration sidecars are
    present. The caller may still run an explicitly sm_86-scoped cohort; a
    cross-arch or sm_120-scoped request receives a typed-unavailable result.
    """
    sm120_cells = [
        (slug, shape_point)
        for slug, shape_point, arch in headline_cohort
        if is_sm120_template_arch(arch)
    ]
    n_sm120 = len(sm120_cells)
    n_total = len(headline_cohort)
    if n_sm120 == 0:
        return PredictorStageRouting(outcome="legacy", n_total=n_total)
    examples = sm120_cells[:3]
    ellipsis = "..." if n_sm120 > 3 else ""
    return PredictorStageRouting(
        outcome="sm120_template_unavailable",
        n_sm120=n_sm120,
        n_total=n_total,
        diagnostic=SM120_TEMPLATE_UNAVAILABLE_REASON,
        detail=(
            f"{n_sm120} sm_120 template cell(s) in cohort "
            f"({examples}{ellipsis}); target_arch={target_arch!r}. "
            "The sm_120 producer, calibration, and comparison paths are "
            "disabled until live per-architecture evidence closes the "
            "research gaps. Run an explicitly sm_86-scoped cohort for the "
            "active lane."
        ),
    )


_SM120_FROZEN_ROSTER_PATH = (
    _REPO_ROOT / "data" / "local_labels"
    / "sm120_cohort_expansion_roster.toml"
)
_SM120_LABEL_ADMITTED_DEFAULT_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
_SM120_DERIVED_COHORT_FLOOR = 30


class Sm120DerivedCohort(NamedTuple):
    """Output of deriving the historical sm_120 cohort artifacts."""

    cohort: list[tuple[str, str, str]]
    manifest: "vcm.CohortManifest"


def _sm120_template_unavailable_result(
    *,
    headline_cohort: list[tuple[str, str, str]],
    target_arch: str | None,
) -> dict:
    """Build the runner's non-scoring sm_120 gate result.

    This report intentionally is not a ``comparison_bundle``: writing one
    would make a disabled architecture look like a scored comparison. The
    caller emits the JSON diagnostic on stderr and returns non-zero, leaving
    historical bundles/sidecars byte-for-byte untouched.
    """
    cells = [
        {
            "slug": slug,
            "shape_point": shape_point,
            "arch": SM120_ARCH,
        }
        for slug, shape_point, arch in headline_cohort
        if is_sm120_template_arch(arch)
    ]
    return unavailable_result(
        operation="run_headline_cohort_predict_compare",
        arch=SM120_ARCH,
        target_arch=target_arch,
        unavailable_cells=cells,
        unavailable_cell_count=len(cells),
        comparison_typed_reason="target_arch_unlabeled_extrapolation",
    )


def _derive_sm120_cohort_from_committed_artifacts(
    *,
    roster_path: Path = _SM120_FROZEN_ROSTER_PATH,
    label_admitted_path: Path = _SM120_LABEL_ADMITTED_DEFAULT_PATH,
) -> Sm120DerivedCohort:
    """Build the active sm_120 cohort from the committed loop
    artifacts: the frozen roster (slug+shape+family+split+arch_gate
    rows) and ``label_admitted.json`` (which roster cells are
    actually admitted as sm_120). The active cohort is the
    intersection: roster cells whose ``(slug, shape_point)`` pair
    appears in ``label_admitted.json`` with ``arch == "sm_120"``.

    Returns ``(cohort, manifest)``: ``cohort`` is a list of
    ``(slug, shape_point, "sm_120")`` triples in roster
    ``candidate_order`` (the deterministic admission order); the
    ``manifest`` is a synthetic ``CohortManifest`` whose
    ``admitted_cell_count`` matches the derived cell count so the
    existing publication-gate floor check works unchanged.

    Raises ``FileNotFoundError`` if either committed artifact is
    missing; raises ``RuntimeError`` if no admitted sm_120 cells
    intersect the roster (an empty cohort is a hard error, not a
    silent below-floor state)."""
    import tomllib  # local import; the loader is only needed
                    # on the sm_120 derived-cohort path.
    if not roster_path.is_file():
        raise FileNotFoundError(
            f"sm_120 frozen roster missing at {roster_path}; "
            f"--target-arch sm_120 requires the committed roster"
        )
    if not label_admitted_path.is_file():
        raise FileNotFoundError(
            f"label_admitted.json missing at {label_admitted_path}; "
            f"--target-arch sm_120 derived-cohort path requires "
            f"the committed label corpus"
        )
    with roster_path.open("rb") as fh:
        roster_data = tomllib.load(fh)
    with label_admitted_path.open("r", encoding="utf-8") as fh:
        label_data = json.load(fh)
    sm120_admitted_keys: dict[tuple[str, str], dict] = {}
    for cell in label_data.get("cells", []) or []:
        if cell.get("arch") != "sm_120":
            continue
        key = (
            str(cell.get("slug", "")),
            str(cell.get("shape_point", "")),
        )
        if not key[0] or not key[1]:
            continue
        sm120_admitted_keys[key] = cell
    roster_rows = roster_data.get("rows", []) or []
    # Sort by candidate_order so the iteration is deterministic
    # and matches the roster's documented admission order.
    roster_rows_sorted = sorted(
        roster_rows,
        key=lambda r: int(r.get("candidate_order", 0)),
    )
    cohort: list[tuple[str, str, str]] = []
    manifest_cells: list[dict] = []
    for row in roster_rows_sorted:
        key = (str(row.get("slug", "")), str(row.get("shape_point", "")))
        if key not in sm120_admitted_keys:
            continue
        cohort.append((key[0], key[1], "sm_120"))
        label_row = sm120_admitted_keys[key]
        identity = {
            "slug": key[0],
            "shape_point": key[1],
            "binary_sha256": str(label_row.get("binary_sha256", "")),
            "kernel_signature_sha256": str(
                label_row.get("kernel_signature_sha256", "")
            ),
            "source_commit": str(label_row.get("source_commit", "")),
            "arch": "sm_120",
            "device": "RTX 5060 Laptop GPU",
        }
        manifest_cells.append({
            "slug": key[0],
            "shape_point": key[1],
            "arch": "sm_120",
            "device": "RTX 5060 Laptop GPU",
            "family_bucket": str(row.get("family_bucket", "")),
            "split_assignment": str(row.get("split", "")),
            "expected_trace_path": str(row.get("trace_json_path", "")),
            "expected_label_path": (
                "data/local_labels/label_admitted.json"
            ),
            "admission_status": "admitted",
            "exclusion_typed_reason": None,
            "identity_tuple": identity,
        })
    if not cohort:
        raise RuntimeError(
            f"derived sm_120 cohort is empty: no roster cell at "
            f"{roster_path.name} is also admitted as sm_120 in "
            f"{label_admitted_path.name}"
        )
    cells_tuple = tuple(
        vcm.CohortManifestCell.from_dict(c) for c in manifest_cells
    )
    manifest = vcm.CohortManifest(
        schema_version=1,
        frozen_at_commit="0" * 40,
        attempted_cell_count=len(cells_tuple),
        admitted_cell_count=len(cells_tuple),
        replacement_history=tuple(),
        cells=cells_tuple,
        raw={},
    )
    return Sm120DerivedCohort(cohort=cohort, manifest=manifest)


class Sm120BundleSetValidation(NamedTuple):
    """Result of validating a sm_120 producer's bundle set against
    the requested sm_120 cohort. ``ok`` is True iff the producer's
    output is bound one-to-one to the cohort, with no missing keys,
    no extra keys, no duplicates, and every bundle carrying
    ``arch == "sm_120"`` and ``cache_layer == "L2"``.

    The four mismatch fields name the offending keys for the
    typed diagnostic; ``ok == False`` iff at least one of them is
    non-empty (or the validation could not extract a key from
    a bundle, recorded in ``unkeyable``)."""

    ok: bool
    missing: tuple[tuple[str, str], ...] = ()
    extra: tuple[tuple[str, str], ...] = ()
    duplicates: tuple[tuple[str, str], ...] = ()
    wrong_arch: tuple[tuple[str, str], ...] = ()
    wrong_cache_layer: tuple[tuple[str, str], ...] = ()
    unkeyable: tuple[str, ...] = ()
    diagnostic: str = ""
    detail: str = ""


_SM120_BUNDLE_SET_MISMATCH_DIAG = (
    "headline_runner_sm120_producer_bundle_set_mismatch"
)


def _bundle_key(bundle: dict) -> tuple[str, str] | None:
    """Extract the ``(slug, shape_point)`` key from a producer
    bundle's ``manifest_pairing_index``. Returns None if either
    field is absent or empty (the producer must always emit a
    complete MPI; absence is itself a bundle-set defect)."""
    mpi = bundle.get("manifest_pairing_index") or {}
    slug = str(mpi.get("slug", ""))
    shape = str(mpi.get("shape_point", ""))
    if not slug or not shape:
        return None
    return (slug, shape)


def _validate_sm120_producer_bundle_set(
    *,
    cohort: list[tuple[str, str, str]],
    bundles: list[dict],
) -> Sm120BundleSetValidation:
    """Verify the sm_120 producer's output is bound one-to-one to
    the requested cohort.

    For each cohort cell ``(slug, shape_point, "sm_120")``, exactly
    one produced bundle MUST exist whose
    ``manifest_pairing_index.slug`` and ``shape_point`` match. Every
    such bundle MUST carry ``manifest_pairing_index.arch == "sm_120"``
    and top-level ``cache_layer == "L2"``. Extra bundles (not in the
    cohort) and duplicates fail the check.

    Field-shape contract: ``arch`` lives inside the
    ``manifest_pairing_index`` (the producer emits it there); but
    ``cache_layer`` lives at the BUNDLE top level (per
    ``cache_predictor.predict_cache_hit_rate``'s output shape, which
    ``comparison_runner._bundle_join_key`` reads as
    ``bundle.get("cache_layer", "L2")``). The default-to-"L2"
    mirrors the comparison-runner's legacy handling: an absent
    top-level ``cache_layer`` is treated as L2 (the cache-layer
    contract of the loop pinned by ``cache_layers = ["L2"]``);
    only an EXPLICIT non-L2 value fails.

    Returns a ``Sm120BundleSetValidation`` whose ``ok`` is True
    iff all four mismatch lists are empty AND no bundle was
    unkeyable. The ``detail`` field formats the offending keys for
    the operator log."""
    cohort_keys = {(s, sp) for s, sp, _ in cohort}
    bundle_keys: list[tuple[str, str] | None] = [
        _bundle_key(b) for b in bundles
    ]
    seen: dict[tuple[str, str], int] = {}
    unkeyable: list[str] = []
    wrong_arch: list[tuple[str, str]] = []
    wrong_layer: list[tuple[str, str]] = []
    for b, k in zip(bundles, bundle_keys):
        if k is None:
            mpi = b.get("manifest_pairing_index") or {}
            unkeyable.append(
                f"bundle with mpi={mpi!r} lacks slug/shape_point"
            )
            continue
        seen[k] = seen.get(k, 0) + 1
        mpi = b.get("manifest_pairing_index") or {}
        if mpi.get("arch") != "sm_120":
            wrong_arch.append(k)
        # Top-level cache_layer with "L2" default, mirroring
        # comparison_runner._bundle_join_key's reading. The
        # producer is allowed to omit the field for L2-default
        # cells; only an explicit non-L2 value fails.
        if b.get("cache_layer", "L2") != "L2":
            wrong_layer.append(k)
    bundle_key_set = set(seen.keys())
    missing = sorted(cohort_keys - bundle_key_set)
    extra = sorted(bundle_key_set - cohort_keys)
    duplicates = sorted(k for k, n in seen.items() if n > 1)
    ok = (
        not missing
        and not extra
        and not duplicates
        and not wrong_arch
        and not wrong_layer
        and not unkeyable
    )
    detail = ""
    if not ok:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        if duplicates:
            parts.append(f"duplicates={duplicates}")
        if wrong_arch:
            parts.append(f"wrong_arch={wrong_arch}")
        if wrong_layer:
            parts.append(f"wrong_cache_layer={wrong_layer}")
        if unkeyable:
            parts.append(f"unkeyable={unkeyable}")
        detail = "; ".join(parts)
    return Sm120BundleSetValidation(
        ok=ok,
        missing=tuple(missing),
        extra=tuple(extra),
        duplicates=tuple(duplicates),
        wrong_arch=tuple(wrong_arch),
        wrong_cache_layer=tuple(wrong_layer),
        unkeyable=tuple(unkeyable),
        diagnostic="" if ok else _SM120_BUNDLE_SET_MISMATCH_DIAG,
        detail=detail,
    )


def _resolve_calibration_evaluation_sidecar_path(
    *,
    explicit_override: Path | None,
    target_arch: str | None,
    repo_root: Path,
) -> Path | None:
    """Resolve which calibration-evaluation sidecar the headline
    runner consumes per the active calibration sidecar selection
    rule:

      1. If ``target_arch`` denotes sm_120 (including device/build aliases),
         raise the typed sm120-template safety failure even when an explicit
         override is supplied. The v4 sidecar remains a historical artifact
         and is never consumed by production.
      2. Else if ``explicit_override`` is non-None, return it verbatim.
      3. Else (``target_arch`` is None, ``sm_86``, or any other
         legacy value), fall back to
         ``data/local_labels/calibration_evaluation_v1.json``
         when it exists, otherwise return None (legacy
         no-calibration-gate behavior).
    """
    if is_sm120_template_arch(target_arch):
        raise Sm120TemplateUnavailable(
            operation="run_headline_cohort_predict_compare.calibration",
            detail=(
                "sm_120 calibration sidecars are retained as historical "
                "evidence and cannot be consumed by the production path"
            ),
        )
    if explicit_override is not None:
        return explicit_override
    data_dir = repo_root / "data" / "local_labels"
    legacy = data_dir / "calibration_evaluation_v1.json"
    return legacy if legacy.is_file() else None


def _import_label_loader():
    """Module accessor for the shared label-loader's
    operational-blinding API. Returns the imported module so
    callers can reference both ``assert_path_is_not_sealed``
    and ``BlindedSealedLabelAccessError`` from a single
    namespace. Wrapped in a helper so test code can monkeypatch
    the source if the loader's import surface ever needs to
    relocate."""
    return _label_loader_mod

# The legacy default for `--cohort-manifest` pointed at the
# sm_86 manifest under
# `data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json`.
# A prior pass flipped 44 broken-trace admitted entries in that
# manifest to `attempted_excluded` to make the committed manifest
# self-consistent on a fresh checkout, which left the legacy
# default with 0 admitted cells (failing the publication floor
# before any prediction work). The current contract REMOVES the
# default entirely: the `--cohort-manifest` flag is now required.
# Operators MUST pass a manifest path explicitly. The sm_120-only
# decoupled framework's mainline path already passes
# `--cohort-manifest <sm_120_path>` explicitly; the sm_86 cohort
# recovery loop on a 3090 Ti host will rebuild the admitted
# entries and the operator can then resume passing the sm_86
# manifest as the explicit argument.
_DEFAULT_COHORT_MANIFEST_PATH: Path | None = None

_DEFAULT_BASE_CONTRACT_PUBLICATION_GRADE_FLOOR = 10
"""Per-arch publication-grade admitted-cell floor that the base
``comparison_contract.toml`` enforces. The runner reads this constant
when no submission-track overlay is supplied; when an overlay is
active, the floor is sourced from
:attr:`SubmissionOverlay.publication_grade_floor_per_arch` instead.

The two-tier separation matters because the base floor is a contract
hard requirement (the AC the comparison-runner already enforces),
while the overlay's per-arch floor MAY differ for a single-arch
submission scope and the submission target (unique_slugs + cells) is
a strictly tighter publication target that lives only under the
overlay."""


SUBMISSION_GATE_NOT_MET_TAG = "submission_gate_not_met"
"""Top-level tag appended to ``publication_grade_tags`` on the
emitted comparison bundle when the submission gate fails (overlay
active, contract floor passed, but unique_slugs or total_cells
below the overlay's paper target). Downstream report generation
reads this tag to suppress publication-grade claims while still
emitting the underlying comparison artifacts."""


def _load_manifest_for_runner_gates(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    label_admitted_path: Path | None = None,
) -> vcm.CohortManifest:
    """Load the cohort manifest with path-reconciliation + the
    identity-bound label-presence check, returning the validated
    :class:`vcm.CohortManifest` object. Used by the runner's
    publication-grade gate stage so the gates see the SAME
    reconciled + label-backed view that
    :func:`_load_admitted_cohort_from_manifest` iterates.

    Every cell that remains admitted post-path-reconciliation is
    cross-referenced against the canonical
    ``data/local_labels/label_admitted.json`` aggregate on the
    5-field identity key; the loader raises
    :class:`vcm.CohortManifestLabelPresenceError` for any admitted
    cell whose identity tuple is absent from the aggregate. This
    closes the trace-only false-positive arm where a cell points
    at an existing aggregate file with no matching row.
    """
    return vcm.validate_manifest_with_path_reconciliation(
        manifest_path,
        repo_root=repo_root,
        label_admitted_path=label_admitted_path,
        enforce_label_presence=True,
    )


def _load_admitted_cohort_from_manifest(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    label_admitted_path: Path | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Load the cohort manifest at ``manifest_path``, reconcile each
    admitted cell's expected_trace_path / expected_label_path against
    the filesystem, enforce the identity-bound label-presence check
    against the canonical label_admitted.json aggregate, then return
    the reconciled admitted cells'
    ``(slug, shape_point, arch)`` triples in declared order. Raises
    the typed validator exceptions on schema / duplicate /
    accounting / identity-lockstep / label-presence drift; consumers
    should let them propagate so the runner aborts loudly.

    Path reconciliation: admitted cells whose `expected_trace_path`
    is absent on disk become `attempted_excluded` with
    `exclusion_typed_reason="trace_missing"`; missing
    `expected_label_path` becomes `label_missing`. Reconciled
    `attempted_excluded` cells are NOT returned in the triple, so
    the headline runner skips them without aborting on a partial
    bundle set.

    Label-presence enforcement: every cell that REMAINS admitted
    post-reconciliation must have its 5-field identity tuple
    present in ``data/local_labels/label_admitted.json``;
    mismatches raise :class:`vcm.CohortManifestLabelPresenceError`.

    The third tuple slot carries the manifest-sourced `arch` so the
    runner's overlay arch-scope check can verify each cell against
    the overlay's `scope.arch_set` without falling back to the
    `_HEADLINE_COHORT_DEFAULT_ARCH` module constant. The
    fallback constant is preserved for callers that import the
    headline runner without a manifest, but it is no longer the
    runner's iteration source.
    """
    manifest = vcm.validate_manifest_with_path_reconciliation(
        manifest_path,
        repo_root=repo_root,
        label_admitted_path=label_admitted_path,
        enforce_label_presence=True,
    )
    return tuple(
        (cell.slug, cell.shape_point, cell.arch)
        for cell in manifest.admitted_cells()
    )


def _emit_manifest_validation_diagnostic(
    exc: vcm.CohortManifestValidationError,
) -> None:
    """Emit a stable, typed stderr diagnostic for any cohort manifest
    validation exception. The runner CLI surface matches the
    validator CLI surface so operator-facing diagnostics are
    consistent across the two entry points.

    The LabelPresence subclass emits
    ``cohort_manifest_label_presence: ({slug!r}, {shape!r},
    arch={arch!r}) has no matching identity-tuple row in
    {path}``. The other typed subclasses
    (SchemaError, DuplicateCellError, AccountingDriftError,
    IdentityLockstepError) emit their existing typed messages
    via ``str(exc)``. The fall-through emits a concise
    ``cohort_manifest_validation_error:`` so an unhandled
    subclass still produces a controlled non-zero diagnostic
    rather than a Python traceback.
    """
    if isinstance(exc, vcm.CohortManifestLabelPresenceError):
        print(
            f"cohort_manifest_label_presence: "
            f"({exc.slug!r}, {exc.shape_point!r}, arch={exc.arch!r}) "
            f"has no matching identity-tuple row in "
            f"{exc.label_admitted_path}",
            file=sys.stderr,
        )
        return
    if isinstance(
        exc,
        (
            vcm.CohortManifestSchemaError,
            vcm.CohortManifestDuplicateCellError,
            vcm.CohortManifestAccountingDriftError,
            vcm.CohortManifestIdentityLockstepError,
        ),
    ):
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return
    print(f"cohort_manifest_validation_error: {exc}", file=sys.stderr)


_DEFAULT_DEVICE = "NVIDIA GeForce RTX 3090 Ti"

# 5-tuple key for the local sm_86 RTX 3090 Ti device-properties
# manifest entry. The producer resolves capacity, slice count,
# and the full provenance dict from this entry instead of
# carrying hard-coded constants, so every emitted bundle records
# the exact geometry it was predicted under.
_LOCAL_MANIFEST_KEY = {
    "arch": "sm_86",
    "sku": "rtx_3090_ti",
    "driver_version": "595.71.05",
    "cuda_toolkit_version": "13.2",
    "nvbit_build": "1.8",
}

# Coarse CTA/SM interleaving: round-robin merge per-threadblock
# access sequences before walking the LRU stack so reuse
# distances reflect concurrent-CTA L2 contention. Enabled here
# on sm_86; the back-compat default in the walker / predictor
# stays False so unit tests and committed bundles without this
# provenance field preserve the legacy sequential per-TB walk.
# The manifest's cta_interleave_model field must agree with this
# choice (round_robin_per_sm when True; sequential_per_tb when
# False); a manifest-vs-producer mismatch fails closed in
# _resolve_local_cache_model_geometry below.
_DEFAULT_CACHE_MODEL_CTA_INTERLEAVE_ENABLED = True


def _resolve_local_cache_model_geometry() -> tuple[
    "device_properties.DevicePropertiesEntry", dict
]:
    """Resolve the local sm_86 RTX 3090 Ti device-properties
    manifest entry and return ``(entry, provenance_dict)``.

    The entry's geometry fields source the predictor's
    ``cache_model_capacity_bytes`` and ``cache_model_num_l2_slices``;
    the provenance dict goes into ``cache_model_geometry_provenance``
    on every emitted bundle. Fails closed if the producer's
    ``_DEFAULT_CACHE_MODEL_CTA_INTERLEAVE_ENABLED`` setting
    disagrees with the manifest's ``cta_interleave_model`` claim,
    because a downstream consumer would otherwise see a bundle
    whose top-level interleave flag contradicts its own
    provenance block.
    """
    entry = device_properties.resolve_entry(**_LOCAL_MANIFEST_KEY)
    interleave_model = entry.cta_interleave_model
    if _DEFAULT_CACHE_MODEL_CTA_INTERLEAVE_ENABLED:
        expected_model = "round_robin_per_sm"
    else:
        expected_model = "sequential_per_tb"
    if interleave_model != expected_model:
        raise RuntimeError(
            f"device-properties manifest cta_interleave_model="
            f"{interleave_model!r} disagrees with producer setting "
            f"cache_model_cta_interleave_enabled="
            f"{_DEFAULT_CACHE_MODEL_CTA_INTERLEAVE_ENABLED!r} "
            f"(expected manifest value: {expected_model!r}). Either "
            f"update the manifest to match the producer, or change "
            f"the producer's interleave setting to match the manifest."
        )
    provenance = entry.to_geometry_provenance(
        num_l2_slices_source="device_properties_manifest",
    )
    return entry, provenance


def _current_producer_build_commit() -> str:
    """Return the full SHA of HEAD in the working tree.

    The bundle's top-level `predictor_build_commit` field records
    this value so the round-closing scanner can distinguish the
    PRODUCER-CODE build commit (this value) from the trace's
    pinned source_commit (recorded in `manifest_pairing_index.source_commit`,
    where it must stay so labels join). The two fields differ when
    the regeneration runs OFFLINE against historical traces (the
    common case in Round A0 onward).
    """
    import subprocess  # local import: avoid a top-level dep for tests that monkey-patch
    result = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


# Historical hard-coded cohort tuple. Preserved here as a fallback /
# documentation surface for older tests that still iterate
# ``_HEADLINE_COHORT_13`` directly. The live cohort the runner
# actually iterates is loaded from
# ``data/sm_86_predictor_accuracy_recovery_v2/cohort_manifest.json``
# via :func:`_load_admitted_cohort_from_manifest`. The fallback
# tuple covers the same 13 cells as the seed manifest (9 admitted +
# 4 attempted_excluded with ``typed_unavailable_emit``); the
# manifest narrows the runner's iteration to the 9 admitted cells.
_HEADLINE_COHORT_13: tuple[tuple[str, str], ...] = (
    ("working_set_capacity_sweep", "trace_l2_accuracy_xl"),
    ("l2_thrash_pointer_chase", "trace_l2_accuracy_xl"),
    ("dram_stream_stride_mix", "trace_l2_accuracy_xl"),
    ("carveout_sensitivity_probe", "trace_l2_accuracy_xl"),
    ("warm_cold_pair_relaunch", "trace_l2_accuracy_xl"),
    ("l2_persisting_window_sweep", "trace_l2_accuracy_xl"),
    ("l2_prefetch_hint_sweep", "trace_l2_accuracy_xl"),
    ("register_spill_local_mem", "trace_l2_accuracy_xl"),
    ("layernorm_fwd_bf16_sm80", "timing_shape"),
    ("rmsnorm_fwd_bf16_sm80", "timing_shape"),
    ("silu_fwd_bf16_sm80", "correctness_shape"),
    ("silu_fwd_bf16_sm80", "timing_shape"),
    ("softmax_safe_bf16_sm80", "timing_shape"),
)

# Implicit arch for the current hard-coded headline cohort. The runner
# emits sm_86 results under workloads/simbench_bw/results/<slug>/rtx_3090_ti/sm_86/
# so the cohort iteration carries sm_86 for every cell. A future
# manifest-driven loader is expected to surface per-cell arch tokens
# explicitly; see comparison_contract_overlay.SubmissionOverlay.arch_set
# for the closed enum the overlay scope-check validates against.
_HEADLINE_COHORT_DEFAULT_ARCH = "sm_86"


class OverlayScopeViolation(Exception):
    """Raised by :func:`_check_overlay_scope_for_cohort` when a cohort
    cell's arch is outside the active overlay's ``scope.arch_set``.

    Carries the typed diagnostic literal
    :data:`comparison_contract_overlay.CELL_ARCH_OUTSIDE_OVERLAY_SCOPE_DIAGNOSTIC`
    so the runner can surface the exact token on stderr before
    returning a non-zero exit. The exception is host-independent;
    the helper raises before any GPU or filesystem state is touched.
    """

    def __init__(
        self,
        *,
        slug: str,
        shape_point: str,
        cell_arch: str,
        overlay_arch_set: tuple[str, ...],
    ) -> None:
        self.typed_diagnostic = (
            cco.CELL_ARCH_OUTSIDE_OVERLAY_SCOPE_DIAGNOSTIC
        )
        self.slug = slug
        self.shape_point = shape_point
        self.cell_arch = cell_arch
        self.overlay_arch_set = overlay_arch_set
        super().__init__(
            f"{self.typed_diagnostic}: cohort cell "
            f"{slug}/{shape_point} has arch={cell_arch!r}, "
            f"which is outside the overlay's arch_set="
            f"{list(overlay_arch_set)!r}"
        )


def _check_overlay_scope_for_cohort(
    overlay: "cco.SubmissionOverlay",
    cohort: "Iterable[tuple[str, str, str]]",
) -> None:
    """Validate every cohort cell's arch against ``overlay.arch_set``.

    ``cohort`` yields ``(slug, shape_point, arch)`` triples. The first
    cell whose arch is not in the overlay's closed arch_set raises
    :class:`OverlayScopeViolation`. The helper is the canonical guard
    the runner calls when ``--contract-overlay`` is provided.

    Validating the FIRST-mismatch (rather than collecting all) keeps
    the diagnostic loud and small; a cohort that admits one sm_120 cell
    almost certainly admits more, but the operator only needs to learn
    the gate fired once.
    """
    arch_set = frozenset(overlay.arch_set)
    for slug, shape_point, arch in cohort:
        if arch not in arch_set:
            raise OverlayScopeViolation(
                slug=slug,
                shape_point=shape_point,
                cell_arch=arch,
                overlay_arch_set=tuple(sorted(arch_set)),
            )


def _apply_submission_overlay_to_contract(
    base_contract: "cr.ComparisonContract",
    overlay: "cco.SubmissionOverlay",
) -> "cr.ComparisonContract":
    """Return a new ``ComparisonContract`` derived from
    ``base_contract`` with the overlay's ``arch_set`` and
    ``cache_layers`` substituted in.

    The base contract describes the project's broader two-arch program
    (``arch_set = ("sm_86", "sm_120")``). The submission-track overlay
    narrows the comparison's admission scope to a single arch + single
    cache layer; this helper is the single place the narrowing
    happens, so the comparison runner sees a contract whose
    ``arch_set``, ``cache_layers``, and downstream coverage gates are
    all consistent with the overlay.

    ``--contract-overlay`` is sufficient on its own: ``--target-arch``
    is NOT a required co-flag (the overlay's narrowed ``arch_set``
    is the canonical signal).

    All other contract fields (epsilon, bootstrap_resamples,
    join_key, ncu_metric_version, per_arch_diagnostics_config, the
    full ``CoverageGate`` block, etc.) carry through unchanged from
    the base contract because those pins are program-level and are
    NOT submission-track-specific.
    """
    import dataclasses
    return dataclasses.replace(
        base_contract,
        arch_set=tuple(overlay.arch_set),
        cache_layers=tuple(overlay.cache_layers),
    )


def _trace_dir_for(slug: str, shape_point: str) -> Path:
    return (
        _RESULTS_ROOT / slug / "rtx_3090_ti" / "sm_86"
        / f"trace_{shape_point}"
    )


def _trace_sidecar_for(slug: str, shape_point: str) -> Path:
    return (
        _RESULTS_ROOT / slug / "rtx_3090_ti" / "sm_86"
        / f"trace_{shape_point}.json"
    )


def _ensure_sector_stream(
    traces_folder: Path,
    *,
    sources: AdmittedTraceSources,
) -> AdmittedSectorStream:
    """Regenerate sector JSONL from the currently admitted snapshots."""
    result = consume_admitted_trace_sources(
        sources=sources,
        output_root=traces_folder,
        write_jsonl=True,
    )
    if not result.schema_admits_all:
        raise TraceAdmissionError(
            "sector_stream_schema_invalid",
            "headline sector-stream producer generated a schema-invalid "
            f"record: {result.schema_first_failure}",
        )
    result.verify_for(sources)
    return result


def _read_manifest_pairing_index(slug: str, shape_point: str) -> dict:
    """Build the predictor's manifest_pairing_index from the trace
    JSON sidecar. Carries the 5-field provenance binding that the
    comparison runner's join key uses (the runner adds arch +
    ncu_metric_version to make the 7-field key)."""
    sidecar = _trace_sidecar_for(slug, shape_point)
    with open(sidecar) as f:
        payload = json.load(f)
    return {
        "slug": str(payload["slug"]),
        "shape_point": str(payload["shape_point"]),
        "binary_sha256": str(payload["binary_sha256"]),
        "source_commit": str(payload["source_commit"]),
        "kernel_signature_sha256": str(payload["kernel_signature_sha256"]),
    }


def _load_launch_parameters_sidecar_for_cell(
    *,
    slug: str,
    shape_point: str,
    expected_identity: dict[str, str],
    arch: str = "sm_86",
) -> tuple[dict[str, int] | None, Path | None, str | None]:
    """Load + semantically validate the per-cell launch-parameter
    sidecar. Returns ``(grid_dim, sidecar_path, sha256_hex)``
    when the sidecar exists at the canonical filename AND its
    embedded identity tuple matches ``expected_identity``.

    Per-arch routing uses the shared
    ``arch_token_policy.sidecar_token_for_label_arch`` mapping:
    ``sm_86`` -> ``sm_86``, ``sm_120`` -> ``sm_120-laptop``.

    Returns ``(None, None, None)`` when the sidecar file is
    missing AND ``arch == "sm_86"`` (legacy fall-through to the
    no-correction path). For any other arch, raises
    :class:`arch_token_policy.SidecarDirectoryMissingError` when
    the per-arch sidecar directory is absent, OR returns
    ``(None, None, None)`` when the directory exists but the
    individual cell's sidecar is missing (an in-progress sweep
    case the operator should populate).

    Raises ``cache_predictor.LaunchParametersSidecarIdentityMismatch``
    when the sidecar exists but its identity disagrees with the
    caller, OR when ``grid_dim`` is malformed."""
    if is_sm120_template_arch(arch):
        raise Sm120TemplateUnavailable(
            operation="run_headline_cohort_predict_compare.launch_sidecar",
            detail="sm_120 predictor-side launch correction is disabled",
        )
    import sys as _sys
    _tools_local_labels = _REPO_ROOT / "tools" / "local_labels"
    if str(_tools_local_labels) not in _sys.path:
        _sys.path.insert(0, str(_tools_local_labels))
    import arch_token_policy as _atp  # type: ignore[import-not-found]
    sidecar_token = _atp.sidecar_token_for_label_arch(arch)
    sidecar_dir = (
        _REPO_ROOT / "data" / "local_labels"
        / "launch_parameters_sidecars" / sidecar_token
    )
    # For non-sm_86 arches, the per-arch sidecar directory MUST exist
    # on disk; a missing directory is a typed-error condition (the
    # operator has not populated sm_120 sidecars yet) rather than a
    # silent fall-through that would skip gridDim correction.
    if arch != "sm_86" and not sidecar_dir.is_dir():
        raise _atp.SidecarDirectoryMissingError(
            f"Launch-parameter sidecar directory missing for arch "
            f"{arch!r}: {sidecar_dir}. The operator-stage sweep must "
            f"populate this directory before predictor cells with "
            f"arch={arch!r} can apply the gridDim correction. The "
            f"sm_86 path admits a legacy no-correction fall-through; "
            f"non-sm_86 arches do not, to avoid silent miscalibration."
        )
    sidecar_path = (
        sidecar_dir
        / f"launch_parameters_sidecar_{slug}__{shape_point}.json"
    )
    if not sidecar_path.is_file():
        return None, None, None
    # Lazy import to avoid a circular import at module-load time
    # (cache_predictor lives in workloads/simbench_bw/, this is
    # in tools/local_labels/).
    import sys
    _cp_dir = _REPO_ROOT / "workloads" / "simbench_bw"
    if str(_cp_dir) not in sys.path:
        sys.path.insert(0, str(_cp_dir))
    import cache_predictor as _cp  # type: ignore[import-not-found]
    grid_dim, sha256_hex = _cp.load_launch_parameters_sidecar(
        sidecar_path=sidecar_path,
        expected_identity=expected_identity,
    )
    return grid_dim, sidecar_path, sha256_hex


def predict_one_cell(
    slug: str,
    shape_point: str,
    output_dir: Path,
    *,
    bootstrap_seed: int = 0,
    variance_precondition_stddev: float | None = None,
) -> Path:
    """Run cache_predictor for one cell and write the bundle to
    ``output_dir/<arch>__<slug>__<shape_point>.json``. Returns the
    bundle's on-disk path.

    The ``bootstrap_seed`` kwarg forwards directly into
    ``cp.predict_cache_hit_rate`` so an operator-specified seed
    affects both the per-cell predictor CIs (here) AND the
    comparison-level CI (in ``main`` via
    ``cr.run_comparison(bootstrap_seed=...)``). Default 0 matches
    the historical pre-threaded behavior for callers that don't
    pass the kwarg.

    The ``variance_precondition_stddev`` kwarg threads the
    cross-cell prediction stddev measured on the admitted-cell
    cohort into the predictor's variance-precondition gate. When
    set + below the documented floor (0.10), the predictor emits
    typed_unavailable with `calibration_variance_precondition_failed`
    per the typed-fail-closed contract. When None (the default),
    no gate behavior; preserves byte-for-byte legacy for callers
    that don't pass the kwarg."""
    traces_folder = _trace_dir_for(slug, shape_point)
    admission = require_trace_admission(traces_folder)
    sector_stream_path = (
        traces_folder / "sector_stream" / "sector_stream.jsonl"
    )
    mpi = _read_manifest_pairing_index(slug, shape_point)
    entry, provenance = _resolve_local_cache_model_geometry()
    # The caller-side identity tuple for the launch-parameter sidecar
    # is sourced from the trace-side manifest_pairing_index (which is
    # itself read from the trace JSON sidecar at
    # `_read_manifest_pairing_index`). The `arch` token is added here
    # because the cohort runner is the natural arch boundary
    # (per-cell trace sidecars do not embed arch; the canonical sm_86
    # value is the orchestrator's responsibility).
    expected_identity = {
        "arch": "sm_86",
        "slug": mpi["slug"],
        "shape_point": mpi["shape_point"],
        "binary_sha256": mpi["binary_sha256"],
        "kernel_signature_sha256": mpi["kernel_signature_sha256"],
    }
    grid_dim, sidecar_path, sidecar_sha256 = (
        _load_launch_parameters_sidecar_for_cell(
            slug=slug,
            shape_point=shape_point,
            expected_identity=expected_identity,
        )
    )
    with open_admitted_trace_sources(
        traces_folder, expected_admission=admission
    ) as sources:
        sector_stream = _ensure_sector_stream(
            traces_folder,
            sources=sources,
        )
        bundle = cp.predict_cache_hit_rate_from_admitted_sources(
            sources=sources,
            sector_stream=sector_stream,
            trace_dir=traces_folder,
            sector_stream_path=sector_stream_path,
            cache_model_capacity_bytes=entry.l2_capacity_bytes,
            manifest_pairing_index=mpi,
            device=_DEFAULT_DEVICE,
            bootstrap_seed=bootstrap_seed,
            cache_model_num_l2_slices=entry.num_l2_slices,
            cache_model_num_l2_slices_source="device_properties_manifest",
            cache_model_cta_interleave_enabled=(
                _DEFAULT_CACHE_MODEL_CTA_INTERLEAVE_ENABLED
            ),
            cache_model_hrd_second_layer_enabled=True,
            cache_model_geometry_provenance=provenance,
            cache_model_variance_precondition_stddev=(
                variance_precondition_stddev
            ),
            grid_dim=grid_dim,
            launch_parameters_sidecar_path=sidecar_path,
            launch_parameters_sidecar_sha256=sidecar_sha256,
            sharing_locality_path=(
                traces_folder / "sharing_locality" / "sharing_locality.json"
            ),
            hrd_profile_path=(
                traces_folder / "hrd_profile" / "hrd_profile.json"
            ),
            launch_parameters_sidecar_identity=(
                expected_identity if sidecar_path is not None else None
            ),
        )
    bundle["predictor_build_commit"] = _current_producer_build_commit()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sm_86__{slug}__{shape_point}.json"
    cp.write_bundle(bundle, out_path)
    return out_path


# ---------------------------------------------------------------------
# L1 diagnostic emission (appendix-only surface; not part of headline)
# ---------------------------------------------------------------------
#
# L1 has two DISTINCT surfaces, not to be confused:
#  (1) The SCORED L1TEX predictor bundles (load-bearing). The headline runner
#      emits per-cell L1TEX bundles whose predicted_hit_rate is CALIBRATED by the
#      fitted L1 head (data/local_labels/sm_86_l1_head.json) and scored by the
#      comparison runner's L1 layer in lockstep with cache_layers = ["L1", "L2"].
#  (2) The L1 diagnostic SIDECAR (appendix-only). A SEPARATE per-cell sidecar in
#      a directory the comparison runner's predictor-bundle discovery never globs,
#      still carrying the typed placeholder `l1_diagnostic_head_pending`. The
#      sidecar is NOT a predictor bundle and never enters the headline geomean;
#      promoting it to carry a scalar L1 prediction is a separate follow-up. The
#      scored path (1) is what makes L1 load-bearing; the sidecar (2) is unchanged.
#
# When `--l1-diagnostic-dir=""` is passed (explicit opt-out), the SIDECAR
# emission is skipped; this mirrors the back-compat opt-out pattern used for
# `--variance-precondition-artifact`. The opt-out is for operators who want
# byte-identical pre-overlay runner behavior; all submission-track flows should
# leave the flag at its default. The opt-out does NOT affect the scored L1TEX
# bundles in (1).

L1_DIAGNOSTIC_SCHEMA_VERSION = 1
L1_DIAGNOSTIC_ARTIFACT_KIND = "l1_diagnostic"
L1_DIAGNOSTIC_HEAD_PENDING_REASON = "l1_diagnostic_head_pending"

# Committed per-cell L1 walker predictions + the fitted L1 calibration head. The
# headline runner sources its scored L1TEX predictor bundles by CALIBRATING the
# walker predictions through the L1 head, in lockstep with the widened
# cache_layers = ["L1", "L2"] contract. Distinct from the l1_diagnostic sidecar:
# these bundles ENTER the comparison runner's L1 layer (scored / typed_unavailable
# via the head), the sidecar never does.
_L1_PREDICTIONS_PATH = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "predicted_l1_per_admitted_cell.json"
)
_L1_HEAD_PATH = _REPO_ROOT / "data" / "local_labels" / "sm_86_l1_head.json"
_L1_FAMILY_BUCKETS_PATH = (
    _REPO_ROOT / "data" / "local_labels" / "sm120_cohort_family_buckets.toml"
)


def _emit_l1tex_bundles_for_cohort(
    headline_cohort: list, bundles_dir: Path,
) -> list[dict]:
    """Build + write per-cell L1TEX predictor bundles for the cohort.

    Sources scalar / typed_unavailable L1 predictions from the committed
    per-cell L1 walker artifact, filtered to the requested cohort, then
    CALIBRATES every scalar row through the fitted L1 head
    (``data/local_labels/sm_86_l1_head.json``) via the shared
    head-application path so the scored values come from the HEAD, not the raw
    walker. Writes each bundle to ``bundles_dir`` under a distinct ``__L1TEX``
    filename so it does not collide with the L2 bundle. The returned bundles are
    appended to the comparison runner's bundle set so the L1 layer is scored
    (scalar cells) or typed-excluded (typed_unavailable cells), never silently
    dropped.

    Fail-closed: if the L1 head is missing/invalid or cannot score a scalar row,
    this RAISES (via the closure module's validation/application). It returns an
    empty list when the committed walker-prediction artifact is absent OR when no
    cohort cell matches (no L1 cells to emit at all).

    Empty-cohort host-independence: when no cohort cell matches, this returns ``[]``
    BEFORE loading / rehydrating the (sklearn-backed) L1 head, so a contract-overlay
    smoke test that monkeypatches the cohort to empty stays reproducible on a host
    without scikit-learn. The fail-closed head load/validate/apply path is reached
    only when there is at least one cohort cell to score.
    """
    if not _L1_PREDICTIONS_PATH.is_file():
        return []
    predictions = json.loads(_L1_PREDICTIONS_PATH.read_text(encoding="utf-8"))
    cohort_keys = {
        (arch, slug, shape_point)
        for slug, shape_point, arch in headline_cohort
        if not is_sm120_template_arch(arch)
    }
    cohort_cells = [
        c for c in predictions.get("cells", [])
        if (
            str(c.get("arch", "")),
            str(c.get("slug", "")),
            str(c.get("shape_point", "")),
        ) in cohort_keys
    ]
    # Empty cohort -> nothing to emit; return BEFORE importing/loading the
    # sklearn-backed L1 head so the empty-cohort path is host-independent
    # (no scikit-learn required). The fail-closed head path below runs only when
    # there is at least one cohort cell to score.
    if not cohort_cells:
        return []
    from emit_l1_closure_artifacts import (
        build_head_calibrated_l1tex_bundles,
        load_and_validate_l1_head,
    )
    # Load + validate the L1 head (fail closed), then calibrate the cohort's
    # walker predictions through it. No raw-walker fallback.
    l1_head = load_and_validate_l1_head(_L1_HEAD_PATH)
    bundles, _provenance, _selected = build_head_calibrated_l1tex_bundles(
        {"cells": cohort_cells},
        l1_head,
        family_buckets_path=_L1_FAMILY_BUCKETS_PATH,
    )
    out: list[dict] = []
    for cell, bundle in zip(cohort_cells, bundles):
        out_path = (
            bundles_dir
            / f"sm_86__{cell['slug']}__{cell['shape_point']}__L1TEX.json"
        )
        out_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        out.append(bundle)
    return out


def _emit_l1_diagnostic_for_cell(
    *,
    slug: str,
    shape_point: str,
    arch: str,
    l1_diagnostic_dir: Path,
    identity_tuple: dict[str, str],
    source_trace_sidecar: Path | None,
) -> Path:
    """Emit an L1 diagnostic sidecar for the cell.

    The sidecar lives at
    ``<l1_diagnostic_dir>/<slug>__<shape_point>.json`` and carries the
    cell's identity tuple + a typed-reason indicating the L1
    diagnostic head is pending. The path is INTENTIONALLY
    distinct from the predictor-bundles directory so the comparison
    runner's bundle discovery cannot pick it up.

    The atomic-write pattern (write-to-temp, fsync, rename) is the
    same the rest of the local-labels driver uses for sidecars.
    """
    l1_diagnostic_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": L1_DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_kind": L1_DIAGNOSTIC_ARTIFACT_KIND,
        "slug": slug,
        "shape_point": shape_point,
        "arch": arch,
        "identity_tuple": dict(identity_tuple),
        "l1_prediction": {
            "result_kind": "typed_unavailable",
            "typed_reason": L1_DIAGNOSTIC_HEAD_PENDING_REASON,
        },
    }
    if source_trace_sidecar is not None:
        payload["source_trace_sidecar"] = (
            str(source_trace_sidecar.relative_to(_REPO_ROOT))
            if _REPO_ROOT in source_trace_sidecar.resolve().parents
            else str(source_trace_sidecar)
        )
    out_path = l1_diagnostic_dir / f"{slug}__{shape_point}.json"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predictor for all 13 cohort cells then run the "
            "comparison and emit comparison_bundle.json."
        ),
    )
    parser.add_argument(
        "--label-admitted-path",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "label_admitted.json",
        help=(
            "Path to label_admitted.json (default: "
            "data/local_labels/label_admitted.json)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output dir for predictor bundles + the final "
            "comparison_bundle.json."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Bootstrap seed (default: 0).",
    )
    parser.add_argument(
        "--target-arch",
        default=None,
        help=(
            "Optional single-arch admission gate. The active production "
            "lane is sm_86; sm_120 and its device/build aliases return a "
            "typed-unavailable result before prediction or comparison."
        ),
    )
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=None,
        help=(
            "Path to the cohort_manifest.json that pins the cohort "
            "the runner iterates. NO default — operators MUST supply "
            "this explicitly. When omitted, the runner emits a typed "
            "diagnostic `cohort_manifest_argument_required` to stderr "
            "and returns 1 (the omission is reported by the runner's "
            "business-logic path, not by argparse, so callers see a "
            "uniform fail-closed shape across all missing-precondition "
            "modes). The sm_120-only decoupled framework's mainline "
            "path passes the sm_120 cohort manifest from the sm_120 "
            "activation bundle producer's sweep dir. The runner "
            "loops over the manifest's "
            "admission_status=='admitted' entries; the legacy "
            "hard-coded _HEADLINE_COHORT_13 tuple has been replaced. "
            "The manifest is strict-validated by "
            "tools/sm_86_predictor_accuracy_recovery_v2/validate_cohort_manifest.py "
            "before iteration."
        ),
    )
    parser.add_argument(
        "--contract-overlay",
        type=Path,
        default=None,
        help=(
            "Optional path to a submission-track overlay TOML (e.g. "
            "workloads/simbench_bw/comparison_contract_sm86_submission.toml). "
            "When provided, the overlay's scope.arch_set is enforced "
            "as an additional cohort gate: any cohort cell whose arch "
            "is outside the overlay's arch_set emits the typed "
            "diagnostic cell_arch_outside_overlay_scope and the runner "
            "exits non-zero. The overlay also supplies the submission "
            "target gates (cohort_paper_target_unique_slugs / "
            "cohort_paper_target_cells) consumed by downstream "
            "publication-grade reports. The base "
            "workloads/simbench_bw/comparison_contract.toml is read "
            "unchanged regardless of the overlay; the overlay narrows "
            "the submission view rather than replacing the broader "
            "two-arch contract."
        ),
    )
    parser.add_argument(
        "--l1-diagnostic-dir",
        # type=str (not type=Path) so the explicit empty-string opt-out
        # can be detected at the boundary; Path("") -> Path(".") would
        # mask the opt-out signal.
        type=str,
        default=None,
        help=(
            "Optional path to the L1 diagnostic sidecar directory "
            "(default: data/sm_86_predictor_accuracy_recovery_v2/l1_diagnostics/). "
            "Per cohort cell the runner writes an L1 diagnostic JSON "
            "to <dir>/<slug>__<shape_point>.json. The directory is "
            "INTENTIONALLY distinct from the predictor-bundles "
            "directory: the comparison runner's bundle discovery "
            "never globs this path, so the L1 diagnostic is "
            "appendix-only -- it does not enter the headline geomean. "
            "Pass --l1-diagnostic-dir='' (empty string) to explicitly "
            "opt out of L1 diagnostic emission (legacy back-compat "
            "for callers that need byte-identical pre-overlay runner "
            "behavior)."
        ),
    )
    parser.add_argument(
        "--calibration-evaluation-sidecar",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to a calibration_evaluation "
            "sidecar used only by the active sm_86 path. The sm_120 "
            "template lane rejects both explicit overrides and default "
            "historical v4 sidecars before they are opened. For sm_86, "
            "the override is honored verbatim; when omitted, the runner "
            "falls back to the historical "
            "data/local_labels/calibration_evaluation_v1.json when it "
            "exists, otherwise None (no calibration gate). Retained "
            "v2/v3/v4 sidecars are never promoted by this runner."
        ),
    )
    parser.add_argument(
        "--variance-precondition-artifact",
        # NOTE: type=str (not type=Path) so the explicit empty-string
        # opt-out can be detected at the boundary. With type=Path,
        # argparse would convert "" to Path(".") and the opt-out check
        # below would never fire. The boundary converts the non-empty
        # string to Path only when the opt-out is not requested.
        type=str,
        default=None,
        help=(
            "Optional path to the post-refinement variance "
            "trajectory artifact (produced by the recovery loop's "
            "variance-precondition adjudication round; default: "
            "data/sm_86_predictor_accuracy_recovery/ac6_variance_trajectory_post_ac4.json "
            "when that file exists). When set, the loader reads "
            "current_stddev and asserts precondition_status; the "
            "value is threaded into cp.predict_cache_hit_rate via "
            "cache_model_variance_precondition_stddev. When set "
            "AND the input stddev falls below the documented 0.10 "
            "floor, the predictor short-circuits to typed-unavailable "
            "with calibration_variance_precondition_failed. Pass "
            "--variance-precondition-artifact='' (empty string) to "
            "explicitly opt out (legacy back-compat for callers "
            "that need byte-identical pre-gate behavior)."
        ),
    )
    args = parser.parse_args(argv)
    if args.target_arch is not None:
        # Normalize device/build spellings at the boundary so aliases such as
        # ``sm120-laptop`` cannot bypass the template gate and ``sm86`` keeps
        # the active mixed-cohort continuation semantics.
        args.target_arch = canonical_arch(args.target_arch)

    # Operational blinding gate: this runner consumes the admitted-
    # label corpus, not the sealed holdout truths. If the operator
    # passes the sealed labels path as --label-admitted-path, the
    # path-refusal helper fails closed BEFORE any disk read or
    # Sealed-label refusal check (operational blinding contract)
    # runs BEFORE any cohort-precondition gate. The sealed-label
    # violation is higher-severity than a missing-flag check; if
    # an operator passes the sealed holdout path as
    # --label-admitted-path AND omits --cohort-manifest, the
    # sealed-label diagnostic must fire first so the violation
    # is reported correctly, not masked by the precondition.
    #
    # The runner's effective mode for the admitted-label corpus is
    # calibration/validation; only the holdout-evaluation runner
    # (a separate entry point under holdout_unblind_context()) may
    # consume the sealed file.
    try:
        _import_label_loader().assert_path_is_not_sealed(
            args.label_admitted_path, "calibration",
        )
    except _import_label_loader().BlindedSealedLabelAccessError as exc:
        print(
            f"sealed_holdout_labels_access_refused: "
            f"--label-admitted-path={args.label_admitted_path} "
            f"resolves to the sealed holdout labels file; "
            f"calibration/validation-mode runners must not open "
            f"the sealed truths. detail: {exc}",
            file=sys.stderr,
        )
        return 1

    # Hard safety boundary before any manifest/label/cohort reads: an explicit
    # sm_120 target is a request for the disabled template lane, not a reason
    # to derive a cohort and accidentally score retained artifacts.
    if is_sm120_template_arch(args.target_arch):
        result = _sm120_template_unavailable_result(
            headline_cohort=[],
            target_arch=args.target_arch,
        )
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    # Missing-precondition fail-closed: the `--cohort-manifest`
    # flag has no default (no silent-broken fallback). When the
    # operator omits it, the runner emits a typed diagnostic
    # `cohort_manifest_argument_required` and returns 1. This
    # supersedes the prior `required=True` argparse contract:
    # argparse no longer exits before the runner reaches its
    # other guard paths (publication gate, etc.), so every
    # missing-precondition has a uniform business-logic
    # fail-closed shape on stderr instead of a mix of argparse
    # SystemExit + runner returns. The sealed-label refusal
    # above runs BEFORE this check so a sealed-path violation
    # is never masked by the missing-cohort-manifest diagnostic.
    if args.cohort_manifest is None and args.target_arch == "sm_120":
        # sm_120 derived-cohort path. Read the committed roster +
        # label_admitted.json instead of requiring a hand-built
        # sm_86-style cohort manifest. The sm_120 activation
        # bundle producer does not emit a cohort manifest, so the
        # operator-facing path under --target-arch sm_120 derives
        # the cohort directly from the loop artifacts that
        # already pin the admitted set.
        try:
            derived = _derive_sm120_cohort_from_committed_artifacts(
                label_admitted_path=args.label_admitted_path,
            )
        except FileNotFoundError as exc:
            print(
                f"sm120_derived_cohort_artifact_missing: {exc}",
                file=sys.stderr,
            )
            return 1
        except RuntimeError as exc:
            print(
                f"sm120_derived_cohort_empty: {exc}",
                file=sys.stderr,
            )
            return 1
        headline_cohort = derived.cohort
        manifest_for_gates = derived.manifest

        def _early_relpath(p: Path) -> str:
            try:
                return str(p.relative_to(_REPO_ROOT))
            except ValueError:
                return str(p)

        print(
            f"=== sm_120 derived-cohort path: "
            f"{len(headline_cohort)} cell(s) from "
            f"{_early_relpath(_SM120_FROZEN_ROSTER_PATH)} "
            f"intersected with admitted sm_120 in "
            f"{_early_relpath(args.label_admitted_path)} ===",
            file=sys.stderr,
        )
    elif args.cohort_manifest is None:
        print(
            "cohort_manifest_argument_required: "
            "--cohort-manifest is required. The headline runner "
            "has no default cohort manifest (the legacy sm_86 "
            "default was removed because its admitted entries point "
            "at gitignored trace artifacts that are absent on a "
            "fresh checkout). Operators MUST supply an explicit "
            "manifest path; the sm_120-only decoupled framework's "
            "mainline path can omit --cohort-manifest when "
            "--target-arch sm_120 is specified (the runner then "
            "derives the active cohort from the committed roster "
            "+ label_admitted.json).",
            file=sys.stderr,
        )
        return 1
    else:
        # Load the cohort from the manifest. The runner's iteration
        # target is the manifest's admitted-cell list; the legacy
        # _HEADLINE_COHORT_13 hard-coded tuple is preserved as a fallback
        # constant but no longer drives main()'s prediction loop. Both
        # loaders pass the user-selected label corpus through so the
        # identity-bound label-presence check validates admitted cells
        # against the SAME aggregate the comparison join consumes.
        try:
            headline_cohort = _load_admitted_cohort_from_manifest(
                args.cohort_manifest,
                label_admitted_path=args.label_admitted_path,
            )
            # Load the same manifest object for the
            # publication-grade gate stage. The cohort-loader
            # returns reconciled (slug, shape_point, arch) triples;
            # the gates need the full CohortManifest so the gate
            # predicates can inspect admitted_cell_count, unique
            # slugs, and per-cell metadata.
            manifest_for_gates = _load_manifest_for_runner_gates(
                args.cohort_manifest,
                label_admitted_path=args.label_admitted_path,
            )
        except vcm.CohortManifestValidationError as exc:
            _emit_manifest_validation_diagnostic(exc)
            return 1

    # Resolve the optional submission-track overlay. When provided,
    # the runner enforces overlay.scope.arch_set as an additional
    # cohort gate. The cohort here is the manifest's admitted set;
    # under the seed manifest every cell is sm_86, so the check is
    # currently a no-op. When the manifest grows to carry sm_120
    # cells, the gate will fire on those.
    if args.contract_overlay is not None:
        overlay = cco.load_overlay(args.contract_overlay)
        # headline_cohort now carries the manifest-sourced arch as
        # its third tuple element; pass it through directly.
        try:
            _check_overlay_scope_for_cohort(overlay, iter(headline_cohort))
        except OverlayScopeViolation as exc:
            print(
                f"=== ABORT: {exc.typed_diagnostic}: cell="
                f"{exc.slug}/{exc.shape_point} arch={exc.cell_arch!r} "
                f"overlay.arch_set={list(exc.overlay_arch_set)!r} ===",
                file=sys.stderr,
            )
            return 1
    else:
        overlay = None

    # Architecture safety gate. The former sm_120 first-class producer path
    # is retired: do not resolve its calibration sidecar, create an output
    # directory, invoke a producer, or call the comparison scorer. An
    # explicitly sm_86-scoped mixed request may continue with its active
    # sm_86 subset; every other sm_120-bearing request returns a typed report.
    routing = _route_predictor_stage(
        headline_cohort=headline_cohort,
        target_arch=args.target_arch,
    )
    if routing.outcome == "sm120_template_unavailable":
        sm86_cohort = [
            (slug, shape, arch)
            for slug, shape, arch in headline_cohort
            if not is_sm120_template_arch(arch)
        ]
        if args.target_arch == "sm_86" and sm86_cohort:
            headline_cohort = sm86_cohort
            print(
                f"=== WARN: {routing.diagnostic}: excluding "
                f"{routing.n_sm120} sm_120 template cell(s); "
                "continuing with explicitly scoped sm_86 cells ===",
                file=sys.stderr,
            )
        else:
            result = _sm120_template_unavailable_result(
                headline_cohort=headline_cohort,
                target_arch=args.target_arch,
            )
            print(
                json.dumps(result, indent=2, sort_keys=True),
                file=sys.stderr,
            )
            return 1

    # Publication-grade two-tier gate stage. The contract floor is a
    # hard fail-fast: when the cohort sits below the per-arch floor,
    # the runner aborts BEFORE prediction to avoid burning predictor
    # time on a sub-floor cohort. The submission gate fires only when
    # an overlay is active (overlay supplies the paper target); when
    # the floor passes but the submission gate fails, the runner
    # CONTINUES and TAGS the emitted bundle with
    # ``submission_gate_not_met`` so downstream report generation
    # can suppress publication-grade claims without losing the
    # comparison artifacts.
    if overlay is not None:
        publication_grade_floor = overlay.publication_grade_floor_per_arch
    else:
        publication_grade_floor = (
            _DEFAULT_BASE_CONTRACT_PUBLICATION_GRADE_FLOOR
        )
    floor_verdict = pg.check_contract_floor(
        manifest_for_gates,
        publication_grade_floor=publication_grade_floor,
    )
    if not floor_verdict.passed:
        print(
            f"=== ABORT: {floor_verdict.typed_diagnostic}: "
            f"admitted_count={floor_verdict.admitted_count} < "
            f"required_floor={floor_verdict.required_floor} ===",
            file=sys.stderr,
        )
        return 1
    if overlay is not None:
        submission_verdict = pg.check_submission_gate(
            manifest_for_gates,
            required_unique_slugs=overlay.cohort_paper_target_unique_slugs,
            required_cells=overlay.cohort_paper_target_cells,
        )
        if not submission_verdict.passed:
            print(
                f"=== WARN: {submission_verdict.typed_diagnostic}: "
                f"unique_slugs={submission_verdict.unique_slugs} "
                f"(required {submission_verdict.required_unique_slugs}); "
                f"cells={submission_verdict.total_cells} "
                f"(required {submission_verdict.required_cells}); "
                f"tagging bundle {SUBMISSION_GATE_NOT_MET_TAG!r}, "
                f"continuing to comparison ===",
                file=sys.stderr,
            )
    else:
        submission_verdict = None
    # The overlay reference is also threaded into the comparison
    # phase: when the overlay is active, the base contract loaded
    # below is narrowed via _apply_submission_overlay_to_contract
    # so cr.run_comparison sees the overlay's arch_set + cache_layers.
    # The overlay's publication-grade gates
    # (cohort_paper_target_unique_slugs /
    # cohort_paper_target_cells) are already consumed by the
    # two-tier publication-gate stage above (check_contract_floor
    # + check_submission_gate); the verdicts feed
    # publication_grade_floor_verdict /
    # publication_grade_submission_verdict on the emitted bundle.

    # Resolve the calibration-evaluation sidecar path via the canonical helper.
    # The helper rejects every sm_120 spelling before opening an explicit or
    # historical v4 sidecar. The active sm_86 path may use the legacy v1
    # sidecar when present and otherwise runs without a calibration gate.
    calibration_sidecar_path = (
        _resolve_calibration_evaluation_sidecar_path(
            explicit_override=args.calibration_evaluation_sidecar,
            target_arch=args.target_arch,
            repo_root=_REPO_ROOT,
        )
    )
    calibration_evaluation_sidecar: dict | None = None
    if calibration_sidecar_path is not None:
        with calibration_sidecar_path.open("r", encoding="utf-8") as fh:
            calibration_evaluation_sidecar = json.load(fh)

    # Resolve the variance-precondition artifact path:
    #   1. caller-supplied path (when --variance-precondition-artifact
    #      is provided);
    #   2. empty string explicitly opts out (legacy back-compat);
    #   3. else the default at
    #      data/sm_86_predictor_accuracy_recovery/ac6_variance_trajectory_post_ac4.json
    #      when it exists;
    #   4. else None (no variance gate).
    variance_stddev: float | None = None
    if args.variance_precondition_artifact is not None:
        # args.variance_precondition_artifact is now type=str (raw
        # argparse string); the empty-string opt-out fires before
        # the Path conversion so Path("") -> Path(".") cannot mask
        # the opt-out signal.
        if args.variance_precondition_artifact == "":
            variance_artifact_path = None
        else:
            variance_artifact_path = Path(
                args.variance_precondition_artifact,
            )
    else:
        # Default resolution order (per R44 contract closing
        # Codex R43-review blocking side issue #3): prefer the
        # v2 trajectory under data/sm_86_predictor_accuracy_recovery_v2/
        # when it exists -- it carries the post-corpus-expansion
        # cross_cell_stddev that the R43 task11 gridDim correction
        # pushed above the documented 0.10 floor. Fall back to the
        # v1 trajectory for back-compat with operators / tests
        # that exercise the pre-v2 path. When neither file exists,
        # the gate is silently disabled (no variance precondition
        # is threaded into the predictor).
        v2_default = (
            _REPO_ROOT / "data"
            / "sm_86_predictor_accuracy_recovery_v2"
            / "ac5_variance_trajectory_post_corpus_expansion.json"
        )
        v1_default = (
            _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery"
            / "ac6_variance_trajectory_post_ac4.json"
        )
        if v2_default.is_file():
            variance_artifact_path = v2_default
        elif v1_default.is_file():
            variance_artifact_path = v1_default
        else:
            variance_artifact_path = None
    if variance_artifact_path is not None:
        # Lazy import: the loader only needed when the orchestrator
        # is in a context where the artifact exists.
        import sys as _sys
        _loader_dir = (
            _REPO_ROOT / "tools" / "sm_86_predictor_accuracy_recovery"
        )
        if str(_loader_dir) not in _sys.path:
            _sys.path.insert(0, str(_loader_dir))
        from variance_precondition_loader import (  # noqa: E402
            load_variance_precondition,
        )
        _vp = load_variance_precondition(variance_artifact_path)
        variance_stddev = _vp.current_stddev

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir = output_dir / "predictor_bundles"

    # Resolve the L1 diagnostic emission directory:
    #   1. caller-supplied path (--l1-diagnostic-dir <path>);
    #   2. empty string explicitly opts out (legacy back-compat);
    #   3. else the default at
    #      data/sm_86_predictor_accuracy_recovery_v2/l1_diagnostics/.
    # The default is INTENTIONALLY outside the --output-dir tree so
    # the comparison runner's bundle discovery cannot reach it even
    # if a caller misconfigures globbing.
    l1_diagnostic_dir: Path | None
    if args.l1_diagnostic_dir is None:
        l1_diagnostic_dir = (
            _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
            / "l1_diagnostics"
        )
    elif args.l1_diagnostic_dir == "":
        l1_diagnostic_dir = None
    else:
        l1_diagnostic_dir = Path(args.l1_diagnostic_dir)

    def _safe_relpath(p: Path) -> str:
        try:
            return str(p.relative_to(_REPO_ROOT))
        except ValueError:
            return str(p)

    manifest_label = (
        _safe_relpath(args.cohort_manifest)
        if args.cohort_manifest is not None
        else "<derived sm_120 cohort from committed artifacts>"
    )
    print(
        f"=== running predictor for {len(headline_cohort)} cohort cells "
        f"(output: {_safe_relpath(bundles_dir)}) "
        f"manifest={manifest_label} ===",
        file=sys.stderr,
    )
    if l1_diagnostic_dir is not None:
        print(
            f"  L1 diagnostic dir: "
            f"{_safe_relpath(l1_diagnostic_dir)}",
            file=sys.stderr,
        )
    else:
        print(
            "  L1 diagnostic dir: (opted out via --l1-diagnostic-dir='')",
            file=sys.stderr,
        )
    bundles: list[dict] = []
    failures: list[tuple[str, str, str]] = []
    l1_diagnostic_paths: list[Path] = []
    # The architecture gate ran before any output/prediction work. At this
    # point only the active sm_86 loop can remain.
    skip_legacy_predictor_loop = False
    if not skip_legacy_predictor_loop:
        for slug, shape_point, arch in headline_cohort:
            try:
                bundle_path = predict_one_cell(
                    slug, shape_point, bundles_dir,
                    bootstrap_seed=args.bootstrap_seed,
                    variance_precondition_stddev=variance_stddev,
                )
                with open(bundle_path) as f:
                    bundle = json.load(f)
                bundles.append(bundle)
                print(
                    f"  {slug}/{shape_point}: result_kind="
                    f"{bundle.get('result_kind')!r}; bundle at "
                    f"{_safe_relpath(bundle_path)}",
                    file=sys.stderr,
                )
                # L1 diagnostic emission: per-cell sidecar that is NEVER
                # discovered by the comparison runner's bundle globbing
                # (the file lives in a directory outside `bundles_dir`).
                if l1_diagnostic_dir is not None:
                    try:
                        identity_tuple = _read_manifest_pairing_index(
                            slug, shape_point,
                        )
                        l1_path = _emit_l1_diagnostic_for_cell(
                            slug=slug,
                            shape_point=shape_point,
                            arch=arch,
                            l1_diagnostic_dir=l1_diagnostic_dir,
                            identity_tuple=identity_tuple,
                            source_trace_sidecar=(
                                _trace_sidecar_for(slug, shape_point)
                            ),
                        )
                        l1_diagnostic_paths.append(l1_path)
                    except _HARD_TRACE_SOURCE_ERRORS:
                        raise
                    except Exception as l1_exc:
                        # An L1 diagnostic emission failure is logged
                        # but does NOT fail the round: the diagnostic is
                        # appendix-only and the headline geomean does
                        # not depend on it. The operator sees the
                        # warning on stderr and can re-emit out of band.
                        print(
                            f"    L1 diagnostic emission failed for "
                            f"{slug}/{shape_point}: {l1_exc}",
                            file=sys.stderr,
                        )
            except _HARD_TRACE_SOURCE_ERRORS:
                raise
            except Exception as exc:
                failures.append((slug, shape_point, str(exc)))
                print(
                    f"  {slug}/{shape_point}: PREDICTOR FAILED: {exc}",
                    file=sys.stderr,
                )

    expected = len(headline_cohort)
    # For the first-class sm_120 path, the producer-fail-closed
    # check is enforced by the producer's non-zero rc handled
    # above and the empty-bundles fallback. The legacy
    # predictor-loop fail-closed check below applies only when
    # the predictor loop actually ran.
    if not skip_legacy_predictor_loop and (failures or len(bundles) != expected):
        print(
            f"\n=== ABORT: predictor produced {len(bundles)}/"
            f"{expected} bundles; refusing to run comparison on a "
            f"partial bundle set (predictor fail-closed) ===",
            file=sys.stderr,
        )
        if failures:
            print("Failed cells:", file=sys.stderr)
            for slug, shape_point, exc in failures:
                print(
                    f"  - {slug}/{shape_point}: {exc}",
                    file=sys.stderr,
                )
        return 1

    # L1 promotion (lockstep with the widened cache_layers = ["L1", "L2"]
    # contract): append per-cell L1TEX predictor bundles so the comparison
    # runner SCORES the L1 layer (scalar cells) or typed-excludes it
    # (typed_unavailable cells) instead of marking every L1 cell
    # join_key_unmatched. sm_86-only legacy path (the L1 walker is sm_86-coupled
    # and the committed L1 predictions are sm_86). Done AFTER the L2
    # fail-closed check so the L2 bundle-count discipline is unaffected.
    if not skip_legacy_predictor_loop:
        l1tex_bundles = _emit_l1tex_bundles_for_cohort(
            headline_cohort, bundles_dir,
        )
        bundles.extend(l1tex_bundles)
        print(
            f"  appended {len(l1tex_bundles)} L1TEX predictor bundles "
            f"(L1 layer now scored alongside L2)",
            file=sys.stderr,
        )

    print(
        f"\n=== loading label corpus from "
        f"{_safe_relpath(args.label_admitted_path)} ===",
        file=sys.stderr,
    )
    cells = cr._load_admitted_label_cells(args.label_admitted_path)
    print(f"  loaded {len(cells)} cells", file=sys.stderr)

    print("\n=== loading comparison contract ===", file=sys.stderr)
    contract_path = _SUITE_ROOT / "comparison_contract.toml"
    contract = cr.load_comparison_contract(contract_path)
    if overlay is not None:
        contract = _apply_submission_overlay_to_contract(contract, overlay)
        print(
            f"  applied submission overlay: arch_set="
            f"{list(contract.arch_set)!r} cache_layers="
            f"{list(contract.cache_layers)!r}",
            file=sys.stderr,
        )

    print("\n=== running comparison ===", file=sys.stderr)
    if calibration_evaluation_sidecar is not None:
        print(
            f"  calibration-evaluation sidecar: "
            f"{_safe_relpath(calibration_sidecar_path)}",
            file=sys.stderr,
        )
    out = cr.run_comparison(
        contract=contract,
        predictor_bundles=bundles,
        label_cells=cells,
        bootstrap_seed=args.bootstrap_seed,
        target_arch=args.target_arch,
        calibration_evaluation_sidecar=calibration_evaluation_sidecar,
    )

    # Expose the publication-grade gate verdicts on the emitted
    # comparison bundle so downstream readers (report generation,
    # cascade-recap scanners, RLCR review tooling) can see whether
    # the runner cleared the contract floor and (when overlay-scoped)
    # the submission gate. The floor verdict is always present; the
    # submission verdict is present only when an overlay was active.
    out["publication_grade_floor_verdict"] = {
        "passed": floor_verdict.passed,
        "admitted_count": floor_verdict.admitted_count,
        "required_floor": floor_verdict.required_floor,
        "typed_diagnostic": floor_verdict.typed_diagnostic,
    }
    if submission_verdict is not None:
        out["publication_grade_submission_verdict"] = {
            "passed": submission_verdict.passed,
            "unique_slugs": submission_verdict.unique_slugs,
            "total_cells": submission_verdict.total_cells,
            "required_unique_slugs": submission_verdict.required_unique_slugs,
            "required_cells": submission_verdict.required_cells,
            "typed_diagnostic": submission_verdict.typed_diagnostic,
        }
        if not submission_verdict.passed:
            tags = list(out.get("publication_grade_tags", []))
            if SUBMISSION_GATE_NOT_MET_TAG not in tags:
                tags.append(SUBMISSION_GATE_NOT_MET_TAG)
            out["publication_grade_tags"] = tags

    output_path = output_dir / "comparison_bundle.json"
    cr.write_bundle(out, output_path)
    print(
        f"\n=== comparison_bundle written: "
        f"{_safe_relpath(output_path)} ===",
        file=sys.stderr,
    )
    print(
        f"  result_kind={out.get('result_kind')!r}",
        file=sys.stderr,
    )
    if out.get("result_kind") == "admitted":
        print(
            f"  geomean={out.get('geomean'):.4f} "
            f"ci=[{out.get('geomean_ci_low'):.4f}, "
            f"{out.get('geomean_ci_high'):.4f}]",
            file=sys.stderr,
        )
    else:
        print(
            f"  typed_reason={out.get('typed_reason')!r}",
            file=sys.stderr,
        )

    sm_86 = (
        out.get("per_arch_diagnostics", {})
        .get("per_arch", {})
        .get("sm_86", {})
    )
    print(
        f"  per_arch.sm_86.headline_count_live="
        f"{sm_86.get('coverage_gate_verdict', {}).get('headline_count_live')}",
        file=sys.stderr,
    )
    print(
        f"  per_arch.sm_86.excluded_reason_counts="
        f"{sm_86.get('excluded_reason_counts')!r}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
