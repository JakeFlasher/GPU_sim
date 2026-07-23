"""Fit and emit the v5 monotonic calibration head.

Loads the admitted label cohort + per-cell predictor bundles (which must
carry the canonical_pre_calibration_features block from the the canonical-features contract backfill)
+ family-bucket assignments, then fits a PerBucketCalibratorEnsemble per the
the model-class hierarchy / the per-bucket minimum-N rule protocol (sklearn IR-on-projection when sklearn is
available; pygam GAM / LightGBM monotone GBM / XGBoost fallback per the
registered hierarchy when sklearn is not).

Emits ``data/local_labels/calibration_head_v5.json`` with full provenance:

  - feature_schema (the 6-feature names + monotonic constraint signs)
  - model_class_per_bucket (the selected class for each non-pooled bucket)
  - monotonic_constraints_by_feature (recorded on USED features only)
  - fold_assignments (deterministic stratified round-robin, seed=20260527)
  - selected_hyperparameters (per-class; for IR there are none beyond
    increasing/out_of_bounds clip)
  - dependency_versions (sklearn / pygam / lightgbm / xgboost as probed)
  - training_cell_ids (per bucket)
  - source_commit (git HEAD at emission time)
  - dependency_availability (full probe result)
  - composite_score_formula (verbatim per the model-class hierarchy)
  - per-bucket residual block (LOOCV residuals + holdout if a split is
    available)
  - cohort_scope (records which cells were ADMITTED into the fit vs
    SKIPPED with a typed reason -- e.g. ``canonical_pre_calibration_features_absent``
    when the backfill could not populate the cell because its sharing_locality
    or hrd_profile sidecar was unstaged on this host)

The cohort the fit admits is fully determined by which bundles are supplied
(via ``--bundle-glob`` OR ``--admitted-bundles-from-ledger``) and which label
cells they join to on ``(arch, slug, shape_point)``. Supplying a pure sm_86
bundle/label corpus yields the active standalone single-arch head. A default
or joint corpus containing any sm_120 row is rejected with a typed-unavailable
result before bundle fitting or artifact emission; sm_120 artifacts remain
historical evidence only.

Each emitted artifact also carries a coarse top-level ``status`` field
(``ok`` when the fit ran, else ``indeterminate_insufficient_data``) alongside
the precise ``fit_status``; ``status`` is the at-a-glance fit outcome and
``fit_status`` retains the specific reason.

CLI:
  # Historical joint/sm_120 invocation: returns typed-unavailable without
  # loading bundles or emitting a head.
  python tools/local_labels/fit_monotonic_calibration_head_v5.py \
    [--label-admitted data/local_labels/label_admitted.json] \
    [--bundle-glob 'data/local_labels/sweeps/sm120_cohort_bundles_*/predictor_bundles/*.json'] \
    [--family-buckets data/local_labels/sm120_cohort_family_buckets.toml] \
    [--output data/local_labels/calibration_head_v5.json]

  # Standalone single-arch head pinned to a regen ledger's admitted set:
  python tools/local_labels/fit_monotonic_calibration_head_v5.py \
    --admitted-bundles-from-ledger data/local_labels/sm86_cohort_regen_ledger.json \
    --output data/local_labels/sm_86_v5_head.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Shared predictor-fit cohort definition: since the controlled refit the
# fitted v5 head trains on EVERY admitted sm_86 cell — the register-cap build
# variants and the formerly pending-refit factorial / re-captured cells are
# folded in (the transitional pending sets are empty). The producer scopes its
# bundle selection AND its label-cell join to this cohort — the SAME
# definition the test/report scoping helpers use, so producer output and
# consumer expectation cannot drift. Sibling register-cap variants share their
# (slug, shape_point) name, so every join in this fitter keys on the CLOSED
# identity (binary_sha256-bearing) and cell identities carry the canonical
# variant_id.
from corpus_cell_identity import (  # noqa: E402
    cell_variant_id,
    is_base_predictor_fit_cell,
)
from workloads.simbench_bw import canonical_features as cf  # noqa: E402
from workloads.simbench_bw import monotonic_calibration_head as mch  # noqa: E402
from tools.local_labels.sm120_template_safety import (  # noqa: E402
    canonical_arch,
    is_sm120_template_arch,
    unavailable_result,
)
from workloads.simbench_bw.comparison_runner import (  # noqa: E402
    _arch_for_device,
)

# Default file paths (overridable via CLI).
DEFAULT_LABEL_ADMITTED = "data/local_labels/label_admitted.json"
DEFAULT_BUNDLE_GLOB = (
    "data/local_labels/sweeps/sm120_cohort_bundles_*/predictor_bundles/*.json"
)
DEFAULT_FAMILY_BUCKETS_TOML = (
    "data/local_labels/sm120_cohort_family_buckets.toml"
)
DEFAULT_OUTPUT = "data/local_labels/calibration_head_v5.json"


def _sm120_template_unavailable_artifact(
    *,
    requested_arch: str | None,
    output_path: Path | None,
    detail: str | None = None,
) -> dict[str, Any]:
    """Return a fit-shaped typed-unavailable report without writing output."""
    result = unavailable_result(
        operation="fit_monotonic_calibration_head_v5",
        arch="sm_120",
        target_arch=requested_arch,
        output_path=None,
        fit_status="sm120_template_unavailable",
        cohort_scope={
            "admitted": [],
            "skipped": [],
            "admitted_count": 0,
            "skipped_count": 0,
        },
        requested_output_path=(str(output_path) if output_path else None),
    )
    if detail is not None:
        result["detail"] = detail
    return result


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_label_cells(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cells = raw.get("cells", [])
    return [dict(c) for c in cells]


def _admitted_bundle_paths_from_ledger(ledger_path: Path) -> list[Path]:
    """Return the predictor-bundle paths of the ledger's ``admitted`` cells.

    The regeneration ledger is the source of truth for WHICH per-cell bundles
    reached admission; it also records each admitted bundle's sha256 under
    ``artifact_sha256.predictor_bundle``. Selecting bundles by the ledger
    (rather than a filesystem glob) pins the fit to the audited admitted set
    and is collision-proof: a filesystem glob over the regen sweep dirs can
    match more than one bundle for the same ``(slug, shape_point)`` key (e.g.
    a re-cascade that left a superseded bundle on disk), and the downstream
    ``(arch, slug, shape_point)`` join would then silently substitute the
    non-admitted candidate. The sha256 check fails closed if the on-disk
    bundle has drifted from the digest the ledger admitted.
    """
    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for cell in led.get("cells", []):
        if cell.get("status") != "admitted":
            continue
        # Scope to the predictor-fit cohort. Since the controlled refit the
        # register-cap build variants + the formerly pending cells are FOLDED
        # IN (the pending sets are empty and the predicate admits variants);
        # the closed-identity bundle join (binary_sha256-bearing) keeps
        # sibling caps collision-free. A future corpus addition re-populates
        # the pending sets and is excluded here until the next refit.
        if not is_base_predictor_fit_cell(cell):
            continue
        rel = cell.get("predictor_bundle_path")
        if not rel:
            raise ValueError(
                "admitted ledger cell is missing predictor_bundle_path: "
                f"{cell.get('arch')}/{cell.get('slug')}/{cell.get('shape_point')}"
            )
        candidate = Path(rel)
        resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"admitted predictor bundle path does not exist on disk: {resolved}"
            )
        expected = (cell.get("artifact_sha256") or {}).get("predictor_bundle")
        if expected:
            actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    "admitted predictor bundle sha256 mismatch for "
                    f"{rel}: ledger={expected} disk={actual}"
                )
        paths.append(resolved)
    return sorted(paths)


def _load_family_buckets(path: Path) -> dict[str, str]:
    """Parse the family-bucket TOML and return a dict mapping raw family ->
    bucket. Pure-Python parse without depending on the tomllib import path
    so this works on stdlib-only setups."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    in_block = False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_block = line == "[family_to_bucket]"
            continue
        if not in_block:
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            # Strip inline `# comment` tail before parsing the value.
            if "#" in v:
                v = v.split("#", 1)[0]
            v = v.strip().strip(",").strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            out[k] = v
    return out


def _slug_to_raw_family(slug: str) -> str:
    """Naive slug -> raw family extraction.

    sm120 cohort slugs follow conventions like
    ``rmsnorm_fwd_bf16_sm80`` (activation), ``gemm_hmma_bf16_sm80``
    (tensor compute), ``carveout_sensitivity_probe`` (memory), etc.
    The raw family is the first 1-2 underscore segments without the
    trailing dtype/arch suffix. Conservative fallback returns ``other`` so
    the the family-bucket validator's closed-enum family-bucket validator's closed-enum stays intact.
    """
    s = slug.lower()
    if "rmsnorm" in s or "layernorm" in s:
        return "rmsnorm" if "rmsnorm" in s else "layernorm"
    if "silu" in s or "swiglu" in s:
        return "swiglu" if "swiglu" in s else "silu"
    if "softmax" in s:
        return "softmax"
    if "cross_entropy" in s:
        return "cross_entropy"
    if "gemm_hmma" in s:
        return "gemm_hmma"
    if "gemm_imma" in s:
        return "gemm_imma"
    if "gemm_sparse" in s:
        return "gemm_sparse"
    if "flash_attn" in s:
        return "attention"
    if "conv2d" in s:
        return "conv2d"
    if "global_atomic" in s:
        return "atomics"
    if "embedding" in s:
        return "embedding"
    if "l2_persisting" in s or "l2_prefetch" in s or "carveout" in s:
        return "cache"
    if "dram_stream" in s or "warm_cold" in s:
        return "memory"
    if "working_set" in s:
        return "memory"
    return "other"


def _resolve_family_bucket(slug: str, family_to_bucket: dict[str, str]) -> str:
    raw = _slug_to_raw_family(slug)
    return family_to_bucket.get(raw, "other")


def _build_cell_records(
    *,
    label_cells: list[dict],
    bundle_paths: list[Path],
    family_to_bucket: dict[str, str],
    split_by_key: dict[tuple, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Join label cells against bundles + canonical block.

    Returns ``(admitted_cells, skipped_cells)`` where each admitted cell is
    a dict with ``identity``, ``arch``, ``family_bucket``, ``features``,
    ``label`` keys, suitable for PerBucketCalibratorEnsemble.fit.
    Skipped cells are dicts with ``identity``, ``arch``, ``slug``,
    ``shape_point``, ``typed_reason``.

    The bundle join keys on the CLOSED identity (arch, slug, shape_point,
    binary_sha256): sibling register-cap build variants share their
    (slug, shape_point) name but each cap is a distinct build with a distinct
    binary_sha256, so the closed key never collapses siblings (and a stale
    bundle for a superseded binary can never join a refreshed label row).
    The emitted cell identity widens with the canonical variant_id so fold
    assignments and downstream rehydration stay collision-free.

    ``split_by_key`` (optional) maps ``(slug, shape_point, variant_id)`` to
    the cohort-split assignment; when provided, every admitted record is
    stamped with its ``split_assignment`` (fail-closed if an sm_86 cell has
    no split row) so the fitter can fit on calibration+validation only and
    the gate emitters can evaluate the held-out cells exactly once.
    """
    bundles_by_key: dict[tuple[str, str, str, str], dict] = {}
    bundle_path_by_key: dict[tuple[str, str, str, str], Path] = {}
    for p in bundle_paths:
        b = json.loads(p.read_text(encoding="utf-8"))
        mpi = b.get("manifest_pairing_index", {})
        # The manifest_pairing_index is the closed 5-tuple with NO arch field,
        # so a real bundle's arch is carried by its canonical top-level
        # `device`. Resolve the arch the SAME way the cohort-comparison joiner
        # does (`comparison_runner._arch_for_device`) so sm_86 and sm_120
        # bundles for a shared (slug, shape_point) key DISTINCTLY and never
        # collide on an empty arch. A bundle whose arch cannot be resolved
        # (no MPI arch + an unrecognized/absent device) keys as "unknown" and
        # will not match any cell — it is skipped as predictor_bundle_absent
        # rather than silently admitted for the wrong arch.
        mpi_arch = str(mpi.get("arch", ""))
        resolved_arch = mpi_arch or _arch_for_device(str(b.get("device", "")))
        key = (
            resolved_arch,
            str(mpi.get("slug", "")),
            str(mpi.get("shape_point", "")),
            str(mpi.get("binary_sha256", "")),
        )
        if key in bundles_by_key:
            # Fail closed: a glob that matches more than one bundle for the
            # same closed-identity cell (e.g. a superseded same-cell bundle
            # from an earlier sweep) would otherwise silently overwrite,
            # training the head on whichever bundle happened to sort last.
            raise ValueError(
                "duplicate predictor bundle key "
                f"{key}: {bundle_path_by_key[key]} and {p} both resolve to the "
                "same (arch, slug, shape_point, binary_sha256). A glob "
                "matched more than one bundle for the same cell; pass "
                "--admitted-bundles-from-ledger or a bundle set with a "
                "unique bundle per cell."
            )
        bundles_by_key[key] = b
        bundle_path_by_key[key] = p
    admitted: list[dict] = []
    skipped: list[dict] = []
    for cell in label_cells:
        arch = str(cell.get("arch", ""))
        slug = str(cell.get("slug", ""))
        shape_point = str(cell.get("shape_point", ""))
        variant_id = cell_variant_id(cell)
        binary_sha256 = str(cell.get("binary_sha256", ""))
        # Bundles are keyed on their resolved arch (MPI arch, else
        # device-derived) + the label row's binary_sha256, so this exact-key
        # lookup is arch- and build-correct by construction: an sm_86 bundle
        # can never be admitted for a sm_120 cell, a sibling register-cap
        # variant can never take its sibling's bundle, and a stale bundle
        # for a superseded binary never joins. A miss means no bundle exists
        # for THIS cell's closed identity -> skip as predictor_bundle_absent.
        bundle = bundles_by_key.get((arch, slug, shape_point, binary_sha256))
        if bundle is None:
            skipped.append({
                "arch": arch,
                "slug": slug,
                "shape_point": shape_point,
                **({"variant_id": variant_id} if variant_id else {}),
                "typed_reason": "predictor_bundle_absent",
                "identity": (arch, slug, shape_point, variant_id),
            })
            continue
        # Propagate typed-unavailable bundles' root-cause reason ahead of
        # the canonical-features-absent symptom check. A bundle with
        # result_kind=="typed_unavailable" carries a typed_reason naming
        # WHY the producer/cache-model could not emit a scalar prediction
        # (e.g., "cache_model_underconfigured"). The downstream
        # canonical_pre_calibration_features_absent symptom is a
        # CONSEQUENCE of that root cause; surfacing it in the skipped
        # row obscures the staging manifest's matching
        # BUNDLE_TYPED_UNAVAILABLE:<reason> classification. Pin the
        # root-cause reason here so the v5 head + downstream gate
        # artifacts + baseline rows all carry the same machine-readable
        # typed-falsification status as the manifest.
        if bundle.get("result_kind") == "typed_unavailable":
            bundle_typed_reason = bundle.get("typed_reason")
            if isinstance(bundle_typed_reason, str) and bundle_typed_reason:
                skipped.append({
                    "arch": arch,
                    "slug": slug,
                    "shape_point": shape_point,
                    **({"variant_id": variant_id} if variant_id else {}),
                    "typed_reason": bundle_typed_reason,
                    "identity": (arch, slug, shape_point, variant_id),
                })
                continue
        features = bundle.get("canonical_pre_calibration_features")
        if features is None:
            skipped.append({
                "arch": arch,
                "slug": slug,
                "shape_point": shape_point,
                **({"variant_id": variant_id} if variant_id else {}),
                "typed_reason": "canonical_pre_calibration_features_absent",
                "identity": (arch, slug, shape_point, variant_id),
            })
            continue
        label_pct = cell.get("cache_hit_rate_l2_sector_pct")
        if label_pct is None:
            skipped.append({
                "arch": arch,
                "slug": slug,
                "shape_point": shape_point,
                **({"variant_id": variant_id} if variant_id else {}),
                "typed_reason": "label_cache_hit_rate_l2_sector_pct_absent",
                "identity": (arch, slug, shape_point, variant_id),
            })
            continue
        family_bucket = _resolve_family_bucket(slug, family_to_bucket)
        record = {
            "identity": (arch, slug, shape_point, variant_id),
            "arch": arch,
            "family_bucket": family_bucket,
            "slug": slug,
            "shape_point": shape_point,
            "features": dict(features),
            "label": float(label_pct) / 100.0,  # convert percent to fraction
        }
        if variant_id:
            record["variant_id"] = variant_id
        if split_by_key is not None:
            split = split_by_key.get((slug, shape_point, variant_id))
            if split is None and arch == "sm_86":
                raise ValueError(
                    "fit_monotonic_calibration_head_v5: admitted sm_86 cell "
                    f"({slug!r}, {shape_point!r}, variant={variant_id!r}) has "
                    "no cohort-split assignment; a split-aware fit cannot "
                    "silently include an unassigned cell"
                )
            if split is not None:
                record["split_assignment"] = split
        admitted.append(record)
    return admitted, skipped


def _score_candidate(
    *,
    cells: list[dict],
    model_class_factory: Any,
    fold_assignments: dict[tuple, int],
) -> dict[str, float]:
    """Compute CV scores for one candidate model class via leave-one-fold-out.

    For each fold k: fit on cells whose fold != k; predict on cells whose
    fold == k; collect (pred, label). Compute geomean rel err / median abs
    err / log-loss over the full collected predictions.
    """
    folds = sorted(set(fold_assignments.values()))
    preds: list[float] = []
    labels: list[float] = []
    for k in folds:
        train_cells = [
            c for c in cells if fold_assignments.get(c["identity"]) != k
        ]
        test_cells = [
            c for c in cells if fold_assignments.get(c["identity"]) == k
        ]
        if not train_cells or not test_cells:
            continue
        ens = mch.PerBucketCalibratorEnsemble(
            model_class_factory=model_class_factory,
        )
        ens.fit(train_cells)
        for c in test_cells:
            try:
                p = ens.predict(c["arch"], c["family_bucket"], c["features"])
            except RuntimeError:
                # Bucket unseen in training fold -- skip; the ensemble has no
                # pooled fallback to route through.
                continue
            preds.append(p)
            labels.append(c["label"])
    primary = mch.geomean_relative_error(preds, labels)
    secondary = mch.median_absolute_error(preds, labels)
    tertiary = mch.log_loss_fractional_bernoulli(preds, labels)
    return {
        "primary": primary,
        "secondary": secondary,
        "tertiary": tertiary,
        "n_cv_predictions": len(preds),
    }


def _model_class_factory_for(name: str) -> Any:
    """Return the constructor for the named model class.

    All 4 registered classes (sklearn IR-on-projection, pygam GAM,
    LightGBM monotone GBM, XGBoost last-resort) have factory shells
    in workloads/simbench_bw/monotonic_calibration_head.py; the secondary
    classes raise typed RuntimeError at fit time when their dependency is
    absent (so the active selector falls through to the next tier).
    """
    factories = {
        "isotonic_regression_on_frozen_projection":
            mch.IsotonicRegressionOnFrozenProjection,
        "monotone_gam_on_six_features": mch.MonotoneGAMOnSixFeatures,
        "monotone_gbm_on_six_features": mch.MonotoneGBMOnSixFeatures,
        "monotone_xgboost_last_resort": mch.MonotoneXGBoostLastResort,
    }
    try:
        return factories[name]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unknown model class {name!r}; the registered hierarchy is "
            f"{tuple(factories.keys())!r}."
        ) from exc


def _load_split_by_key(cohort_split_path: Path) -> dict[tuple, str]:
    """Load ``cohort_split.json`` into a ``(slug, shape_point, variant_id)
    -> split`` map (variant_id is the canonical ``register_cap_<n>`` token,
    ``""`` for a base cell)."""
    payload = json.loads(cohort_split_path.read_text(encoding="utf-8"))
    out: dict[tuple, str] = {}
    for split_name, block in payload.get("splits", {}).items():
        for c in block.get("cells", []):
            out[(
                str(c.get("slug", "")),
                str(c.get("shape_point", "")),
                cell_variant_id(c),
            )] = split_name
    return out


def fit_and_emit_v5(
    *,
    label_admitted_path: Path,
    bundle_paths: list[Path],
    family_buckets_path: Path,
    output_path: Path,
    arch: str | None = None,
    cohort_split_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full v5 fit + emission. Returns the artifact dict.

    When ``arch`` is set, the admitted-label cohort is restricted to cells of
    that architecture BEFORE the bundle join. This is required for a STANDALONE
    single-arch head: the per-cell bundles' manifest_pairing_index is a closed
    5-field identity with NO arch key, so every bundle keys as arch=='' and the
    empty-arch slug+shape_point fallback in _build_cell_records would otherwise
    match a bundle to a same-slug/shape cell of a DIFFERENT arch (the cohort
    reuses the same workload slugs across arches). Filtering the label cohort
    to ``arch`` keeps the fit single-arch and prevents that cross-arch mix.
    """
    requested_arch = (
        canonical_arch(arch) if arch is not None else None
    )
    if is_sm120_template_arch(requested_arch):
        return _sm120_template_unavailable_artifact(
            requested_arch=requested_arch,
            output_path=output_path,
            detail=(
                "sm_120 calibration fitting is disabled; retained v5 "
                "artifacts are historical evidence only"
            ),
        )

    label_cells = _load_label_cells(label_admitted_path)
    if requested_arch is None and any(
        is_sm120_template_arch(c.get("arch")) for c in label_cells
    ):
        # ``arch=None`` historically meant a joint fit. A joint cohort that
        # includes sm_120 would train/emit a cross-architecture head, so stop
        # after the minimum label inspection and before bundle loading or fit.
        return _sm120_template_unavailable_artifact(
            requested_arch=None,
            output_path=output_path,
            detail=(
                "joint/default v5 fitting is disabled when the admitted label "
                "cohort contains sm_120 cells; pass a pure sm_86 corpus with "
                "--arch sm_86"
            ),
        )
    if requested_arch:
        label_cells = [
            c for c in label_cells
            if canonical_arch(c.get("arch")) == requested_arch
        ]
    # Scope the sm_86 label-cell join to the base predictor-fit cohort (no
    # register-cap variant, no pending-refit factorial cell), matching the
    # ledger-bundle scope above so the standalone head's cohort equals the
    # scoped ledger set. The register-cap-variant / pending-refit boundary is an
    # sm_86 corpus concept, so for other arches the `--arch` filter above is the
    # complete scope; the ledger-bundle scope is the load-bearing exclusion in
    # every mode (a cell with no scoped bundle is skipped by the join), and this
    # is the defense-in-depth label-side mirror.
    if requested_arch == "sm_86":
        label_cells = [c for c in label_cells if is_base_predictor_fit_cell(c)]

    def _apply_single_arch_scope(art: dict) -> dict:
        """Tag a head emitted under ``--arch`` as a STANDALONE single-arch head.

        A standalone head is fit independently from ONE architecture's bundles;
        it is NOT a per-arch split of the joint multi-arch head. The
        cross-artifact consistency checks that bind per-arch SPLIT heads to the
        joint head -- the per-arch admitted_count sum equalling the joint head's
        admitted_count, and source_v5_head_commit pinning to the joint head's
        source_commit -- do not apply to a standalone head and key off this
        marker to exempt it.
        """
        if requested_arch:
            art["arch"] = requested_arch
            art["head_scope"] = "standalone_single_arch"
        return art

    family_to_bucket = _load_family_buckets(family_buckets_path)
    split_by_key = (
        _load_split_by_key(cohort_split_path)
        if cohort_split_path is not None else None
    )
    admitted, skipped = _build_cell_records(
        label_cells=label_cells,
        bundle_paths=bundle_paths,
        family_to_bucket=family_to_bucket,
        split_by_key=split_by_key,
    )
    # Split-aware fit scope: when a cohort split is supplied, the model
    # selection CV and the final ensemble fit run on the
    # calibration+validation subset ONLY; the sealed holdout cells are
    # recorded in cohort_scope.admitted (features + labels +
    # split_assignment="holdout") so the gate emitters can evaluate them
    # exactly once against the cal+val-fitted head, but they NEVER enter
    # the fit or the fold assignments.
    if split_by_key is not None:
        fit_cells = [
            c for c in admitted if c.get("split_assignment") != "holdout"
        ]
        if admitted and not fit_cells:
            raise ValueError(
                "fit_monotonic_calibration_head_v5: every admitted cell is "
                "holdout-assigned; a split-aware fit needs a non-empty "
                "calibration+validation subset"
            )
    else:
        fit_cells = admitted
    dep_probe = mch.probe_dependency_availability()
    dependency_versions_flat = {
        e["package"]: e.get("version")
        for e in dep_probe.values()
    }
    if not admitted:
        # Emit a still-schema-valid artifact recording the empty-fit state so
        # the operator can see the cohort-scope diagnostic without re-running.
        artifact = {
            "schema_version": 1,
            "calibration_head_version": 5,
            "source_commit": _git_head_sha(),
            "dependency_availability": dep_probe,
            "dependency_versions": dependency_versions_flat,
            "composite_score_formula": dict(mch.COMPOSITE_SCORE_FORMULA),
            "feature_schema": {
                "canonical_feature_names": list(cf.CANONICAL_FEATURE_NAMES),
                "monotonic_constraints_by_feature": dict(
                    cf.MONOTONIC_CONSTRAINT_SIGNS,
                ),
            },
            "fold_assignments": {},
            "model_class_per_bucket": {},
            "monotonic_constraints_by_feature": {},
            "selected_hyperparameters": {},
            "training_cell_ids": {},
            "per_bucket_provenance": {
                "by_bucket": {},
                "pooled_fallback_present": False,
                "pooled_used_features": [],
            },
            "cohort_scope": {
                "admitted": [],
                "skipped": [
                    {k: v for k, v in s.items() if k != "identity"}
                    for s in skipped
                ],
                "admitted_count": 0,
                "skipped_count": len(skipped),
            },
            "fit_status": "empty_cohort_no_admitted_cells",
            "status": "indeterminate_insufficient_data",
        }
        output_path.write_text(
            json.dumps(_apply_single_arch_scope(artifact), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    # Probe deps + select active model class. When no candidate from the
    # registered hierarchy is importable on this host, emit a still-schema-valid
    # indeterminate artifact rather than raising (so review environments
    # without sklearn / pygam / lightgbm / xgboost do not crash on the
    # default regeneration path).
    dep_avail = dep_probe
    try:
        active_class = mch.select_active_model_class(
            dependency_availability=dep_avail,
        )
    except (RuntimeError, NotImplementedError) as exc:
        artifact = {
            "schema_version": 1,
            "calibration_head_version": 5,
            "source_commit": _git_head_sha(),
            "dependency_availability": dep_probe,
            "dependency_versions": dependency_versions_flat,
            "composite_score_formula": dict(mch.COMPOSITE_SCORE_FORMULA),
            "feature_schema": {
                "canonical_feature_names": list(cf.CANONICAL_FEATURE_NAMES),
                "monotonic_constraints_by_feature": dict(
                    cf.MONOTONIC_CONSTRAINT_SIGNS,
                ),
            },
            "fold_assignments": {},
            "model_class_per_bucket": {},
            "monotonic_constraints_by_feature": {},
            "selected_hyperparameters": {},
            "training_cell_ids": {},
            "per_bucket_provenance": {
                "by_bucket": {},
                "pooled_fallback_present": False,
                "pooled_used_features": [],
            },
            "cohort_scope": {
                "admitted": [],
                "skipped": [
                    {k: v for k, v in s.items() if k != "identity"}
                    for s in skipped
                ] + [
                    {k: v for k, v in {**c, "typed_reason": "deferred_dependencies_absent"}.items() if k not in ("identity", "features")}
                    for c in admitted
                ],
                "admitted_count": 0,
                "skipped_count": len(skipped) + len(admitted),
            },
            "fit_status": "indeterminate_dependencies_absent",
            "status": "indeterminate_insufficient_data",
            "dependency_absence_reason": str(exc),
        }
        output_path.write_text(
            json.dumps(_apply_single_arch_scope(artifact), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact
    # Build fold assignments per the deterministic stratified round-robin.
    identities = [c["identity"] for c in fit_cells]
    strata = [(c["arch"], c["family_bucket"]) for c in fit_cells]
    fold_assignments, fold_decision = mch.deterministic_fold_assignments(
        identities, strata,
    )
    # Iterate over EVERY available registered candidate; score each via
    # the composite protocol; collect skip reasons for unavailable
    # candidates. The winning class is selected by
    # composite_score_select_simpler over the full candidate list.
    candidates: list[dict[str, Any]] = []
    skipped_candidates: list[dict[str, Any]] = []
    for entry in mch.MODEL_CLASS_HIERARCHY:
        cname = entry["name"]
        if not dep_avail.get(cname, {}).get("available"):
            skipped_candidates.append({
                "name": cname,
                "skip_reason": "dependency_unavailable",
                "dependency_package": entry["dependency_package"],
            })
            continue
        try:
            candidate_factory = _model_class_factory_for(cname)
        except NotImplementedError as exc:
            skipped_candidates.append({
                "name": cname,
                "skip_reason": "factory_unavailable",
                "detail": str(exc),
            })
            continue
        try:
            cv_scores = _score_candidate(
                cells=fit_cells,
                model_class_factory=candidate_factory,
                fold_assignments=fold_assignments,
            )
        except (RuntimeError, ImportError, Exception) as exc:
            # Catch fit-time dependency / model failures as structured skip
            # records so the composite-scoring step sees a consistent
            # candidate list even when some classes fail mid-fit.
            skipped_candidates.append({
                "name": cname,
                "skip_reason": "scoring_failure",
                "detail": str(exc),
            })
            continue
        candidates.append({
            "name": cname,
            "parameter_count": entry["parameter_count"],
            "primary": cv_scores["primary"],
            "secondary": cv_scores["secondary"],
            "tertiary": cv_scores["tertiary"],
            "n_cv_predictions": cv_scores["n_cv_predictions"],
        })
    if not candidates:
        raise RuntimeError(
            "no model class from the registered hierarchy completed CV "
            f"scoring; skip_records={skipped_candidates!r}"
        )
    selected = mch.composite_score_select_simpler(candidates=candidates)
    # Final fit on the full admitted set using the SELECTED class's factory
    # (which may differ from selected["name"] when composite_score_select_simpler
    # picks a simpler candidate within the 2% relative tiebreaker window).
    factory = _model_class_factory_for(selected["name"])
    ensemble = mch.PerBucketCalibratorEnsemble(model_class_factory=factory)
    ensemble.fit(fit_cells)
    provenance = ensemble.provenance_summary()
    # Build per-bucket model_class_per_bucket + monotonic_constraints_by_feature.
    model_class_per_bucket: dict[str, str] = {}
    monotonic_by_bucket: dict[str, dict[str, int]] = {}
    head_proto = factory()
    used_features = getattr(
        head_proto, "used_features", cf.CANONICAL_FEATURE_NAMES,
    )
    constraints_for_used = {
        n: cf.MONOTONIC_CONSTRAINT_SIGNS[n] for n in used_features
    }
    for bucket_key, entry in provenance["by_bucket"].items():
        model_class_per_bucket[bucket_key] = (
            selected["name"] if entry["kind"] == "per_bucket"
            else f"{selected['name']}__pooled_fallback_with_one_hot"
        )
        if entry["kind"] == "pooled_fallback":
            monotonic_by_bucket[bucket_key] = dict(constraints_for_used)
            monotonic_by_bucket[bucket_key]["family_bucket_one_hot"] = 0
        else:
            monotonic_by_bucket[bucket_key] = dict(constraints_for_used)
    artifact = {
        "schema_version": 1,
        "calibration_head_version": 5,
        "source_commit": _git_head_sha(),
        "dependency_availability": dep_avail,
        "dependency_versions": dependency_versions_flat,
        "composite_score_formula": dict(mch.COMPOSITE_SCORE_FORMULA),
        "feature_schema": {
            "canonical_feature_names": list(cf.CANONICAL_FEATURE_NAMES),
            "monotonic_constraints_by_feature": dict(
                cf.MONOTONIC_CONSTRAINT_SIGNS,
            ),
        },
        "fold_assignments": {
            # Serialize tuple keys as repr strings for JSON.
            repr(list(ident)): fold for ident, fold in fold_assignments.items()
        },
        "fold_construction": fold_decision,
        "model_class_per_bucket": model_class_per_bucket,
        "monotonic_constraints_by_feature": monotonic_by_bucket,
        "candidate_scores": candidates,
        "skipped_candidates": skipped_candidates,
        "selected_hyperparameters": {
            selected["name"]: {
                "increasing": True,
                "out_of_bounds": "clip",
                "projection_weights": list(mch.FROZEN_PROJECTION_WEIGHTS),
                "projection_features": list(mch.FROZEN_PROJECTION_FEATURES),
            },
        },
        "selected_class": {
            "name": selected["name"],
            "parameter_count": selected["parameter_count"],
            "primary": selected["primary"],
            "secondary": selected["secondary"],
            "tertiary": selected["tertiary"],
            "selection_reason": selected["selection_reason"],
        },
        "training_cell_ids": {
            bucket_key: entry["training_cell_ids"]
            for bucket_key, entry in provenance["by_bucket"].items()
        },
        "per_bucket_provenance": provenance,
        "status": "ok",
        "cohort_scope": {
            "admitted": [
                {
                    "arch": c["arch"],
                    "slug": c["slug"],
                    "shape_point": c["shape_point"],
                    "family_bucket": c["family_bucket"],
                    "label": c["label"],
                    "features": dict(c["features"]),
                    **(
                        {"variant_id": c["variant_id"]}
                        if c.get("variant_id") else {}
                    ),
                    **(
                        {"split_assignment": c["split_assignment"]}
                        if c.get("split_assignment") else {}
                    ),
                }
                for c in admitted
            ],
            "skipped": [
                {k: v for k, v in s.items() if k != "identity"}
                for s in skipped
            ],
            "admitted_count": len(admitted),
            "skipped_count": len(skipped),
        },
        # Split-aware fit scope: when a cohort split was supplied, the model
        # selection CV + the final ensemble fit ran on the
        # calibration+validation subset only; the sealed holdout cells are
        # present in cohort_scope.admitted (split_assignment="holdout") for
        # the gate emitters' exactly-once holdout evaluation but never
        # entered the fit.
        "fit_scope": {
            "split_aware": cohort_split_path is not None,
            "cohort_split_path": (
                str(cohort_split_path) if cohort_split_path is not None
                else None
            ),
            "fit_splits": (
                ["calibration", "validation"]
                if cohort_split_path is not None else None
            ),
            "n_fit": len(fit_cells),
            "n_holdout_excluded_from_fit": len(admitted) - len(fit_cells),
        },
        "fit_status": "ok",
    }
    output_path.write_text(
        json.dumps(_apply_single_arch_scope(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-admitted", default=DEFAULT_LABEL_ADMITTED,
    )
    parser.add_argument(
        "--bundle-glob", default=DEFAULT_BUNDLE_GLOB,
    )
    parser.add_argument(
        "--admitted-bundles-from-ledger", default=None,
        help=(
            "Path to a per-cell regeneration ledger. When set, the fitter "
            "uses EXACTLY the predictor_bundle_path entries of status=='admitted' "
            "cells (sha256-verified against the ledger's "
            "artifact_sha256.predictor_bundle) and IGNORES --bundle-glob. This "
            "pins the fit to the audited admitted set and is collision-proof "
            "against a glob that matches a superseded same-(slug,shape) bundle."
        ),
    )
    parser.add_argument(
        "--family-buckets", default=DEFAULT_FAMILY_BUCKETS_TOML,
    )
    parser.add_argument(
        "--arch", default=None,
        help=(
            "Restrict the admitted-label cohort to this architecture (e.g. "
            "sm_86) before the bundle join, producing a STANDALONE single-arch "
            "head. Required when the supplied bundles carry an arch-less "
            "manifest_pairing_index, to prevent the empty-arch slug/shape "
            "fallback from matching a same-slug cell of another arch."
        ),
    )
    parser.add_argument(
        "--cohort-split", default=None,
        help=(
            "Path to cohort_split.json. When set, the fit becomes "
            "split-aware: model selection CV + the final ensemble fit run "
            "on the calibration+validation subset only, and the sealed "
            "holdout cells are recorded in cohort_scope.admitted "
            "(split_assignment='holdout') for the gate emitters' "
            "exactly-once holdout evaluation."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.arch is not None:
        args.arch = canonical_arch(args.arch)
    if is_sm120_template_arch(args.arch):
        result = _sm120_template_unavailable_artifact(
            requested_arch=args.arch,
            output_path=Path(args.output),
            detail=(
                "sm_120 calibration fitting is disabled; no bundle or label "
                "inputs were loaded"
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        print(
            f"{result['typed_diagnostic']}: {result['typed_reason']}",
            file=sys.stderr,
        )
        return 2
    if args.arch is None:
        # Probe the selected label corpus before resolving a ledger or glob.
        # A joint/default sm_120 cohort must not even read predictor bundles
        # on its way to the typed-unavailable result.
        label_probe = _load_label_cells(Path(args.label_admitted))
        if any(
            is_sm120_template_arch(c.get("arch")) for c in label_probe
        ):
            result = _sm120_template_unavailable_artifact(
                requested_arch=None,
                output_path=Path(args.output),
                detail=(
                    "joint/default v5 fitting is disabled when the admitted "
                    "label cohort contains sm_120 cells; no bundles were "
                    "loaded"
                ),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            print(
                f"{result['typed_diagnostic']}: {result['typed_reason']}",
                file=sys.stderr,
            )
            return 2
    if args.admitted_bundles_from_ledger:
        bundle_paths = _admitted_bundle_paths_from_ledger(
            Path(args.admitted_bundles_from_ledger)
        )
    else:
        bundle_paths = sorted(Path(p) for p in glob.glob(args.bundle_glob))
    artifact = fit_and_emit_v5(
        label_admitted_path=Path(args.label_admitted),
        bundle_paths=bundle_paths,
        family_buckets_path=Path(args.family_buckets),
        output_path=Path(args.output),
        arch=args.arch,
        cohort_split_path=(
            Path(args.cohort_split) if args.cohort_split else None
        ),
    )
    if artifact.get("result_kind") == "typed_unavailable":
        print(json.dumps(artifact, indent=2, sort_keys=True))
        print(
            f"{artifact['typed_diagnostic']}: {artifact['typed_reason']}",
            file=sys.stderr,
        )
        return 2
    print(
        f"[fit_v5] status={artifact['fit_status']} admitted={artifact['cohort_scope']['admitted_count']} "
        f"skipped={artifact['cohort_scope']['skipped_count']} -> {args.output}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
