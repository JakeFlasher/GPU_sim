"""Per-cell predictor-bundle build entry point implementing the
two-stage admission cascade required by the same-74 closure plan.

The cascade is production-active for ``sm_86`` only. ``sm_120`` remains a
retained evidence/template lane: this entry point returns a typed-unavailable
report before scanning or regenerating any bundle for that architecture.

Stage 1 — legacy_match attempt: scan
``data/local_labels/sweeps/**/predictor_bundles/*.json`` for a
bundle whose ``manifest_pairing_index.{slug, shape_point,
binary_sha256, source_commit, kernel_signature_sha256}`` 5-tuple
matches the current ``label_admitted.json`` entry for the requested
cell AND whose ``predictor_build_commit`` resolves to a tree
at-or-after the v5 pin commit ``b1646ef7``. When such a bundle
exists, the ledger entry records
``bundle_admission_path = "legacy_match"`` (no regeneration).

Stage 2 — fresh_regen fallback: when Stage 1 fails (no identity
match OR ``predictor_build_commit < b1646ef7`` OR schema invalid
OR forbidden-digest-key present), invoke the walker via
``workloads.simbench_bw.cache_predictor.predict_cache_hit_rate``
to emit a fresh bundle under
``data/local_labels/sweeps/sm86_cohort_regen_<UTC>/comparison/predictor_bundles/<arch>__<slug>__<shape_point>.json``.
The ledger entry records ``bundle_admission_path = "fresh_regen"``.

This module is the per-cell surface called by the preflight matrix's
``REBUILD_BUNDLE_LOCAL`` proposed action. A bulk-cohort runner is
out of scope for this entry point; for the bulk path, the staging
manifest in conjunction with ``regenerate_v5_artifacts.py`` is the
operator surface.

Forbidden modeled-hit keys (``hrd_profile``, ``hrd_profile_path``,
``hit_count``, ``hit_rate``, ``model_hit``, ``modeled_hit_rate``,
``predicted_hit_rate``) MUST NOT appear in the bundle's
``input_sidecar_digests``; the walker's
``CircularPredictorInputRejected`` exception path enforces this on
the fresh_regen lane, and the legacy-match scan rejects any
candidate that already carries one (defect token
``cache_predictor_circular_predictor_input_rejected``).

Identity vs producer-provenance split:
* ``manifest_pairing_index`` carries the CELL IDENTITY 5-tuple
  matching ``label_admitted.json``.
* ``predictor_build_commit`` carries PRODUCER PROVENANCE (the tree
  the walker module was loaded from).
The two fields MUST remain separate; a bundle whose MPI
``source_commit`` equals ``predictor_build_commit`` blindly is a
defect.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Shared mixed-arity build-variant identity helper (sibling import; the SAME
# normalizer the aggregator / selector / merge / sidecar-regenerator use, so
# base-vs-variant disambiguation never drifts across the bundle cascade).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from corpus_cell_identity import cell_variant_id  # noqa: E402
try:
    from tools.local_labels.sm120_template_safety import (  # noqa: E402
        SM120_TEMPLATE_UNAVAILABLE_REASON,
        is_sm120_template_arch,
        unavailable_result,
    )
except ModuleNotFoundError:  # direct script execution from tools/local_labels
    from sm120_template_safety import (  # noqa: E402
        SM120_TEMPLATE_UNAVAILABLE_REASON,
        is_sm120_template_arch,
        unavailable_result,
    )


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABEL_ADMITTED = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
# Recursive scan pattern matching the plan's loader convention.
# A previous implementation used one-level globs (`sweeps/*/comparison/...`
# and `sweeps/*/predictor_bundles/...`) which silently skipped bundles
# under deeper nesting (e.g., a sweep_run_id with an additional sub
# layout). The recursive pattern below uses ``Path.rglob`` against the
# sweeps root so any depth of intermediate directory is covered.
_SWEEPS_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"
_BUNDLE_GLOB_PATTERN = "predictor_bundles/*.json"
_V5_PIN_COMMIT = "b1646ef7"
_FRESH_REGEN_ROOT = _REPO_ROOT / "data" / "local_labels" / "sweeps"

_FORBIDDEN_DIGEST_KEYS: tuple[str, ...] = (
    "hrd_profile",
    "hrd_profile_path",
    "hit_count",
    "hit_rate",
    "model_hit",
    "modeled_hit_rate",
    "predicted_hit_rate",
)

# Canonical pre-calibration feature names that an admitted bundle's
# ``canonical_pre_calibration_features`` block must carry, each as a
# finite float in the per-feature range pinned by
# ``workloads/simbench_bw/schemas/cache_predictor.schema.json``. The
# cascade's Stage 1 gate enforces presence + finiteness; the
# canonical schema validator enforces per-field range bounds.
_CANONICAL_FEATURE_KEYS: tuple[str, ...] = (
    "raw_prediction",
    "footprint_ratio",
    "long_rd_tail_mass",
    "intra_lane_reuse",
    "global_share",
    "gridDim_density",
)

_CACHE_PREDICTOR_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "cache_predictor.schema.json"
)

_SHARING_LOCALITY_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "sharing_locality.schema.json"
)

_HRD_PROFILE_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "hrd_profile.schema.json"
)

_SECTOR_STREAM_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "sector_stream.schema.json"
)

_DIVERGENCE_PROXIES_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "divergence_proxies.schema.json"
)

_ORDER_PROJECTIONS_SCHEMA_PATH = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "schemas"
    / "order_projections.schema.json"
)

# L2 cache capacity used by the walker for the modeled-hit rule.
# Mirrors `_HRD_CACHE_INPUTS` in
# `tools/local_labels/regenerate_v5_trace_sidecars.py` and
# `tools/hrd_profiler/cache_model_spec.py::resolve_cache_model_spec`:
# sm_86 = 6 MiB (RTX 3090 Ti); sm_120 = 32 MiB (RTX 5060 laptop).
_L2_CAPACITY_BYTES_BY_ARCH: dict[str, int] = {
    "sm_86": 6 * 1024 * 1024,
    "sm_120": 32 * 1024 * 1024,
}

# Canonical device names written into the predictor bundle's top-level
# `device` field. The walker stores this verbatim and downstream the
# cohort-comparison joiner derives the arch from it via
# `comparison_runner._DEVICE_TO_ARCH` (the inverse of this map): a
# canonical name resolves to its arch, an arch token resolves to
# `"unknown"` and the bundle fails the arch join. The keys/values here
# are the inverse of `_DEVICE_TO_ARCH`; the round-trip is pinned by a
# regression test so the two maps cannot drift.
_CANONICAL_DEVICE_BY_ARCH: dict[str, str] = {
    "sm_86": "NVIDIA GeForce RTX 3090 Ti",
    "sm_120": "NVIDIA GeForce RTX 5060 Laptop GPU",
}


def _label_5tuple_for_cell(
    *,
    arch: str,
    slug: str,
    shape_point: str,
    variant_id: str | None = None,
    label_admitted_path: Path = _LABEL_ADMITTED,
) -> dict[str, str] | None:
    """Return the cell-identity 5-tuple from ``label_admitted.json``
    for the requested cell, or ``None`` when no matching row exists.

    ``variant_id`` disambiguates build-variant siblings of the same
    (arch, slug, shape_point): None matches the base cell (no
    ``variant_id`` field); a canonical ``register_cap_<n>`` matches
    that variant row, so its DISTINCT ``binary_sha256`` flows into
    the returned 5-tuple and the Stage-2 walker emits a bundle whose
    identity differs from sibling caps.
    """
    if not label_admitted_path.is_file():
        return None
    try:
        payload = json.loads(label_admitted_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    want_variant = variant_id or ""
    for cell in payload.get("cells", []):
        if not (
            cell.get("arch") == arch
            and cell.get("slug") == slug
            and cell.get("shape_point") == shape_point
        ):
            continue
        if cell_variant_id(cell) != want_variant:
            continue
        required = (
            "slug",
            "shape_point",
            "binary_sha256",
            "source_commit",
            "kernel_signature_sha256",
        )
        if not all(isinstance(cell.get(k), str) and cell[k] for k in required):
            return None
        return {k: cell[k] for k in required}
    return None


def _git_head_sha() -> str | None:
    """Return the current git HEAD SHA-1. Returns None when git is
    unavailable or HEAD is detached without a SHA (rare). Used by
    ``fresh_regen_via_walker`` to stamp ``predictor_build_commit``
    on the emitted bundle so the ledger's admission gate can
    verify the producer-bearing tree.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    if len(sha) != 40 or not all(
        c in "0123456789abcdef" for c in sha
    ):
        return None
    return sha


def _git_is_ancestor(commit: str) -> bool:
    """Return True when ``commit`` is reachable in the repo history
    (a merge-base / ancestor check against HEAD). Returns False when
    git is unavailable or the ref is missing.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "merge-base", "--is-ancestor",
             commit, "HEAD"],
            capture_output=True, text=True,
        )
    except (FileNotFoundError, OSError):
        return False
    return proc.returncode == 0


def _commit_is_after_pin(
    candidate_commit: str, pin_commit: str = _V5_PIN_COMMIT,
) -> bool:
    """Return True when ``candidate_commit`` resolves to a tree
    at-or-after the ``pin_commit``. The check is asymmetric: the
    pin is an ancestor of the candidate, OR the candidate IS the
    pin (i.e., the pin is also at-or-after itself).
    """
    if not isinstance(candidate_commit, str) or not candidate_commit:
        return False
    if candidate_commit.startswith("0" * 8):
        return False
    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "merge-base", "--is-ancestor",
             pin_commit, candidate_commit],
            capture_output=True, text=True,
        )
    except (FileNotFoundError, OSError):
        return False
    return proc.returncode == 0


def _has_forbidden_digest_key(payload: dict) -> str | None:
    """Return the first forbidden digest key present in
    ``input_sidecar_digests``, or ``None`` when none are present.
    """
    digests = payload.get("input_sidecar_digests")
    if not isinstance(digests, dict):
        return None
    for key in _FORBIDDEN_DIGEST_KEYS:
        if key in digests:
            return key
    return None


def _bundle_5tuple(payload: dict) -> dict[str, str] | None:
    mpi = payload.get("manifest_pairing_index")
    if not isinstance(mpi, dict):
        return None
    keys = (
        "slug",
        "shape_point",
        "binary_sha256",
        "source_commit",
        "kernel_signature_sha256",
    )
    out: dict[str, str] = {}
    for k in keys:
        val = mpi.get(k)
        if not isinstance(val, str) or not val:
            return None
        out[k] = val
    return out


def _enumerate_candidate_bundles() -> list[Path]:
    """Walk every predictor bundle under
    ``data/local_labels/sweeps/`` recursively and return the paths
    in sorted order for deterministic Stage 1 ordering.

    The plan pins
    ``data/local_labels/sweeps/**/predictor_bundles/*.json`` as the
    canonical scan pattern. ``Path.rglob`` traverses every
    descendant directory so deeply-nested sweep run roots are
    covered without enumerating a precomputed list of layouts.
    """
    if not _SWEEPS_ROOT.is_dir():
        return []
    return sorted(_SWEEPS_ROOT.rglob(_BUNDLE_GLOB_PATTERN))


def _validate_canonical_features_block(payload: dict) -> str | None:
    """Return ``None`` when the bundle's
    ``canonical_pre_calibration_features`` block carries the six
    required features as finite floats in their schema-pinned
    ranges, else a typed reject token.

    The cascade's Stage 1 enforces presence + finiteness inline
    so a feature-absent or feature-out-of-range candidate is
    rejected before admission. Range bounds are pinned per
    feature by the canonical
    ``cache_predictor.schema.json::canonical_pre_calibration_features``.
    """
    features = payload.get("canonical_pre_calibration_features")
    if not isinstance(features, dict):
        return "canonical_pre_calibration_features_absent"
    missing = [k for k in _CANONICAL_FEATURE_KEYS if k not in features]
    if missing:
        return (
            "canonical_pre_calibration_features_missing_keys:"
            + ",".join(missing)
        )
    for key in _CANONICAL_FEATURE_KEYS:
        val = features[key]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return f"canonical_pre_calibration_features_nonscalar:{key}"
        try:
            f = float(val)
        except (TypeError, ValueError):
            return f"canonical_pre_calibration_features_unparseable:{key}"
        if f != f or f in (float("inf"), float("-inf")):
            return f"canonical_pre_calibration_features_nonfinite:{key}"
    return None


def _validate_bundle_against_schema(payload: dict) -> str | None:
    """Validate the candidate against
    ``workloads/simbench_bw/schemas/cache_predictor.schema.json``
    via ``jsonschema.Draft202012Validator``. Returns ``None`` when
    the candidate admits, else the first validation-error message
    truncated to 256 chars.
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — repo pins jsonschema
        return "jsonschema_unavailable"
    if not _CACHE_PREDICTOR_SCHEMA_PATH.is_file():
        return "cache_predictor_schema_missing"
    try:
        schema = json.loads(
            _CACHE_PREDICTOR_SCHEMA_PATH.read_text(encoding="utf-8"),
        )
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        return (
            f"cache_predictor_schema_load_failed:"
            f"{type(exc).__name__}:{exc!s}"
        )[:256]
    if errors:
        first = errors[0]
        return (
            "cache_predictor_schema_invalid:"
            + ".".join(str(p) for p in first.absolute_path)
            + ":" + first.message
        )[:256]
    return None


def _evaluate_candidate(
    *,
    payload: dict,
    pin_commit: str,
) -> dict[str, Any]:
    """Apply the schema + canonical-feature + forbidden-digest +
    producer-commit gates to one already-identity-matched
    candidate. Each gate fires in cascade order; the first failure
    short-circuits.

    Returns:
    * ``admitted``: bool — passed every gate.
    * ``typed_failure_reason``: str | None — populated when the
      candidate is rejected.
    * ``predictor_build_commit``: str | None — the producer commit
      when present.
    """
    schema_reason = _validate_bundle_against_schema(payload)
    if schema_reason is not None:
        return {
            "admitted": False,
            "typed_failure_reason": schema_reason,
            "predictor_build_commit": None,
        }
    feature_reason = _validate_canonical_features_block(payload)
    if feature_reason is not None:
        return {
            "admitted": False,
            "typed_failure_reason": feature_reason,
            "predictor_build_commit": None,
        }
    forbidden = _has_forbidden_digest_key(payload)
    if forbidden is not None:
        return {
            "admitted": False,
            "typed_failure_reason": (
                "cache_predictor_circular_predictor_input_rejected:"
                + forbidden
            ),
            "predictor_build_commit": None,
        }
    producer_commit = payload.get("predictor_build_commit")
    if not isinstance(producer_commit, str) or not producer_commit:
        producer_commit = payload.get("regeneration_build_commit")
    if not isinstance(producer_commit, str) or not producer_commit:
        return {
            "admitted": False,
            "typed_failure_reason": "predictor_build_commit_missing",
            "predictor_build_commit": None,
        }
    if not _commit_is_after_pin(producer_commit, pin_commit):
        return {
            "admitted": False,
            "typed_failure_reason": "predictor_build_commit_before_pin",
            "predictor_build_commit": producer_commit,
        }
    return {
        "admitted": True,
        "typed_failure_reason": None,
        "predictor_build_commit": producer_commit,
    }


def legacy_match_scan(
    *,
    arch: str,
    slug: str,
    shape_point: str,
    label_5tuple: dict[str, str],
    pin_commit: str = _V5_PIN_COMMIT,
) -> dict[str, Any]:
    """Stage 1 of the cascade. Walk every bundle under
    ``data/local_labels/sweeps/**/predictor_bundles/*.json``,
    classify each into one of three buckets, and admit the first
    candidate that passes every gate.

    The three buckets:

    1. **Full 5-tuple match**: ``manifest_pairing_index`` 5-tuple
       byte-equal to the label. These candidates run through the
       forbidden-digest-key + producer-commit gates; admission
       requires ALL gates to pass.
    2. **Stale-MPI quarantine candidates**: same
       slug+shape_point but mismatched ``binary_sha256`` /
       ``source_commit`` / ``kernel_signature_sha256``. The
       bundle exists on disk and was emitted before the cell's
       identity was refreshed (e.g., before a bulk NCU
       re-capture sweep); admitting it would silently fold
       stale evidence into the cohort, so the cell stays at
       ``legacy_bundle_quarantined`` and the operator-facing
       count surfaces as a queued artifact for cleanup.
    3. **Unrelated**: different slug or different shape_point.
       Filtered out at scan time; not counted.

    Returns a dict with:

    * ``found``: bool — at least one identity-matching (bucket 1)
      bundle existed.
    * ``admitted``: bool — at least one bucket-1 candidate
      passed every gate.
    * ``path``: str | None — repo-relative path to the admitted
      bundle (None on failure).
    * ``typed_failure_reason``: str | None — typed reject reason.
      Priority order: full-match bundle-gate failure >
      legacy_bundle_quarantined > no_legacy_5tuple_match.
    * ``rejected_candidates``: list[dict] — per-bucket-1
      diagnostics (each carries `path`, `typed_failure_reason`,
      `predictor_build_commit`).
    * ``quarantined_candidates``: list[dict] — per-bucket-2
      diagnostics (each carries `path`, `diff_fields` listing
      which 5-tuple fields drifted).
    """
    if is_sm120_template_arch(arch):
        # Do not even scan historical bundles: a matching artifact would
        # otherwise become a live/scorable prediction through Stage 1.
        return {
            "found": False,
            "admitted": False,
            "path": None,
            "typed_failure_reason": SM120_TEMPLATE_UNAVAILABLE_REASON,
            "rejected_candidates": [],
            "quarantined_candidates": [],
            "status": "typed_unavailable",
            "result_kind": "typed_unavailable",
        }

    candidates = _enumerate_candidate_bundles()
    identity_matches: list[tuple[Path, dict]] = []
    quarantined: list[dict[str, Any]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        candidate_5tuple = _bundle_5tuple(payload)
        if candidate_5tuple is None:
            continue
        if candidate_5tuple == label_5tuple:
            identity_matches.append((path, payload))
            continue
        # Quarantine arm: same slug+shape but mismatched MPI.
        # Other-slug / other-shape candidates are unrelated and
        # filtered out.
        if (
            candidate_5tuple["slug"] == label_5tuple["slug"]
            and candidate_5tuple["shape_point"]
                == label_5tuple["shape_point"]
        ):
            diff_fields = [
                k for k in (
                    "binary_sha256",
                    "source_commit",
                    "kernel_signature_sha256",
                )
                if candidate_5tuple[k] != label_5tuple[k]
            ]
            try:
                rel_path = str(path.relative_to(_REPO_ROOT))
            except ValueError:
                rel_path = str(path)
            quarantined.append({
                "path": rel_path,
                "diff_fields": diff_fields,
            })
    if not identity_matches:
        if quarantined:
            diff_summary = ",".join(
                sorted({f for q in quarantined for f in q["diff_fields"]})
            )
            return {
                "found": False,
                "admitted": False,
                "path": None,
                "typed_failure_reason": (
                    f"legacy_bundle_quarantined:"
                    f"{len(quarantined)}:{diff_summary}"
                )[:512],
                "rejected_candidates": [],
                "quarantined_candidates": quarantined,
            }
        return {
            "found": False,
            "admitted": False,
            "path": None,
            "typed_failure_reason": "no_legacy_5tuple_match",
            "rejected_candidates": [],
            "quarantined_candidates": [],
        }
    rejected: list[dict[str, Any]] = []
    for path, payload in identity_matches:
        evaluation = _evaluate_candidate(payload=payload, pin_commit=pin_commit)
        rel_path = str(path.relative_to(_REPO_ROOT))
        if evaluation["admitted"]:
            return {
                "found": True,
                "admitted": True,
                "path": rel_path,
                "typed_failure_reason": None,
                "predictor_build_commit": evaluation["predictor_build_commit"],
                "rejected_candidates": rejected,
                "quarantined_candidates": quarantined,
            }
        rejected.append({
            "path": rel_path,
            "typed_failure_reason": evaluation["typed_failure_reason"],
            "predictor_build_commit": evaluation["predictor_build_commit"],
        })
    aggregate_reason = "all_legacy_candidates_rejected:" + ",".join(
        e["typed_failure_reason"] for e in rejected
    )
    return {
        "found": True,
        "admitted": False,
        "path": None,
        "typed_failure_reason": aggregate_reason[:512],
        "rejected_candidates": rejected,
        "quarantined_candidates": quarantined,
    }


def _resolve_trace_dir(
    *, arch: str, slug: str, shape_point: str, variant_id: str | None = None,
) -> Path | None:
    """Return the on-disk trace_dir for the cell, or None when
    ``dynamic_trace.pb`` does not exist under either workload root.

    ``variant_id`` is additive: None resolves the canonical base
    ``.../<arch>/trace_<shape_point>/`` dir (byte-identical legacy
    behavior); a canonical ``register_cap_<n>`` resolves the
    register-cap variant subtree
    ``.../<arch>/variants/<variant_id>/trace_<shape_point>/`` the
    variant producer writes per cap.
    """
    device_dir = {"sm_86": "rtx_3090_ti", "sm_120": "rtx_5060_laptop"}.get(
        arch, "unknown",
    )
    leaf = (
        Path(f"trace_{shape_point}") if variant_id is None
        else Path("variants") / variant_id / f"trace_{shape_point}"
    )
    for source in ("simbench_bw", "portable_lcd"):
        candidate = (
            _REPO_ROOT
            / "workloads"
            / source
            / "results"
            / slug
            / device_dir
            / arch
            / leaf
        )
        if (candidate / "dynamic_trace.pb").is_file():
            return candidate
    return None


def _required_sidecars_present(trace_dir: Path) -> dict[str, bool]:
    """Return per-sidecar presence flags for the four canonical
    sidecars an admitted cell requires: sharing_locality,
    hrd_profile, sector_stream.jsonl, AND the sibling
    ``sector_stream/manifest_pairing_index.json`` (identity-binding
    file alongside the record-level JSONL).
    """
    return {
        "sharing_locality": (
            trace_dir / "sharing_locality" / "sharing_locality.json"
        ).is_file(),
        "hrd_profile": (
            trace_dir / "hrd_profile" / "hrd_profile.json"
        ).is_file(),
        "sector_stream": (
            trace_dir / "sector_stream" / "sector_stream.jsonl"
        ).is_file(),
        "sector_stream_mpi": (
            trace_dir / "sector_stream" / "manifest_pairing_index.json"
        ).is_file(),
    }


def _validate_sector_stream_jsonl_records(
    *, jsonl_path: Path,
) -> str | None:
    """Validate every non-empty record in
    ``<trace_dir>/sector_stream/sector_stream.jsonl`` against
    ``workloads/simbench_bw/schemas/sector_stream.schema.json``
    via ``jsonschema.Draft202012Validator``.

    The producer schema-validates each line at emission time
    (``tools/sector_stream/consume_trace.py``). A stale,
    hand-copied, or malformed JSONL already on disk skips that
    producer-side gate when fresh_regen runs against it; the
    walker's canonical-feature loader then parses the records
    without validation. This per-line gate closes that hole
    before the walker is invoked.

    Returns ``None`` on full admit, else the first typed reject
    reason of the form
    ``sector_stream_schema_invalid:line_<n>:<path>:<message>``
    (or ``sector_stream_schema_unreadable:line_<n>:...`` /
    ``sector_stream_record_not_object:line_<n>`` / a load-
    failure token). The line number counts only non-empty
    lines; blank or whitespace-only lines do not advance it
    so a malformed record is identified by its record
    position (mirroring how the producer emits one record per
    non-blank line).
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — repo pins jsonschema
        return "sector_stream_jsonschema_unavailable"
    if not _SECTOR_STREAM_SCHEMA_PATH.is_file():
        return "sector_stream_schema_missing"
    try:
        schema = json.loads(
            _SECTOR_STREAM_SCHEMA_PATH.read_text(encoding="utf-8"),
        )
        validator = jsonschema.Draft202012Validator(schema)
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        return (
            f"sector_stream_schema_load_failed:"
            f"{type(exc).__name__}:{exc!s}"
        )[:256]
    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except OSError as exc:
        return (
            f"sector_stream_jsonl_unreadable:"
            f"{type(exc).__name__}:{exc!s}"
        )[:256]
    record_index = 0
    for raw_line in text.split("\n"):
        if not raw_line.strip():
            continue
        record_index += 1
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            return (
                f"sector_stream_schema_unreadable:line_{record_index}:"
                f"{exc.msg!s}"
            )[:256]
        if not isinstance(record, dict):
            return f"sector_stream_record_not_object:line_{record_index}"
        errors = sorted(
            validator.iter_errors(record), key=lambda e: e.path,
        )
        if errors:
            first = errors[0]
            return (
                f"sector_stream_schema_invalid:line_{record_index}:"
                + ".".join(str(p) for p in first.absolute_path)
                + ":" + first.message
            )[:256]
    return None


def _validate_sidecar_against_schema(
    *, sidecar_path: Path, schema_path: Path, producer_label: str,
) -> str | None:
    """Validate a sidecar JSON file against its canonical schema
    via ``jsonschema.Draft202012Validator``. Returns ``None`` when
    the sidecar admits, else a typed reject token of the form
    ``<producer_label>_schema_invalid:<path>:<message>``.
    """
    try:
        import jsonschema  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover — repo pins jsonschema
        return f"{producer_label}_jsonschema_unavailable"
    if not schema_path.is_file():
        return f"{producer_label}_schema_missing"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        return (
            f"{producer_label}_schema_load_failed:"
            f"{type(exc).__name__}:{exc!s}"
        )[:256]
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        first = errors[0]
        return (
            f"{producer_label}_schema_invalid:"
            + ".".join(str(p) for p in first.absolute_path)
            + ":" + first.message
        )[:256]
    return None


def _validate_sidecar_mpi_matches_label(
    *, sidecar_path: Path, label_5tuple: dict[str, str],
    producer_label: str,
) -> str | None:
    """Return ``None`` when the sidecar's embedded
    ``manifest_pairing_index`` block carries every label-5tuple
    field with a matching value, else a typed reject token.

    Each per-producer sidecar carries an embedded MPI block (the
    sharing_locality classifier copies the upstream MPI verbatim;
    the HRD producer fills its MPI from the supplied identity).
    Any drift means a stale or swapped sidecar would feed the
    walker with the wrong-cell identity, so the gate must fire
    before fresh_regen invokes the walker.
    """
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            f"{producer_label}_mpi_unreadable:"
            f"{type(exc).__name__}:{exc!s}"
        )[:256]
    if not isinstance(payload, dict):
        return f"{producer_label}_mpi_not_object"
    mpi = payload.get("manifest_pairing_index")
    if not isinstance(mpi, dict):
        return f"{producer_label}_mpi_missing"
    for key in (
        "slug",
        "shape_point",
        "binary_sha256",
        "source_commit",
        "kernel_signature_sha256",
    ):
        expected = label_5tuple.get(key)
        if mpi.get(key) != expected:
            return (
                f"{producer_label}_mpi_field_mismatch:{key}:"
                f"sidecar={mpi.get(key)!r} label={expected!r}"
            )[:256]
    return None


def _sector_stream_mpi_matches_label(
    *, trace_dir: Path, label_5tuple: dict[str, str],
) -> str | None:
    """Return ``None`` when the sector_stream MPI sibling's payload
    carries every field of ``label_5tuple`` with matching values,
    else a typed reject token.

    The sector_stream sibling carries identity verbatim; a
    mismatch is a hard cell-identity defect under the plan's
    sidecar identity-binding rule.
    """
    sibling = trace_dir / "sector_stream" / "manifest_pairing_index.json"
    try:
        payload = json.loads(sibling.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            f"sector_stream_mpi_unreadable:{type(exc).__name__}:{exc!s}"
        )[:256]
    if not isinstance(payload, dict):
        return "sector_stream_mpi_not_object"
    for key, expected in label_5tuple.items():
        if payload.get(key) != expected:
            return (
                f"sector_stream_mpi_field_mismatch:{key}:"
                f"sibling={payload.get(key)!r} label={expected!r}"
            )[:256]
    return None


def fresh_regen_via_walker(
    *,
    arch: str,
    slug: str,
    shape_point: str,
    label_5tuple: dict[str, str],
    variant_id: str | None = None,
    fresh_regen_run_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stage 2 of the cascade.

    Resolves the trace dir + required sidecars; if any prerequisite
    is missing, returns a typed-failure dict. When prerequisites
    are met (and ``dry_run`` is False), invokes
    ``workloads.simbench_bw.cache_predictor.predict_cache_hit_rate``
    and writes the resulting bundle under
    ``data/local_labels/sweeps/sm86_cohort_regen_<run_id>/comparison/predictor_bundles/<arch>__<slug>__<shape>[__<variant_id>].json``.

    ``variant_id`` routes the trace-dir resolution to the per-cap
    variant subtree AND widens the output filename with a
    ``__<variant_id>`` segment so sibling caps write DISTINCT bundle
    files that cannot overwrite one another.
    """
    if is_sm120_template_arch(arch):
        return unavailable_result(
            operation="fresh_regen_via_walker",
            arch=arch,
            cell_key=(
                f"{arch}/{slug}/{shape_point}"
                if variant_id is None
                else f"{arch}/{slug}/{shape_point}/{variant_id}"
            ),
            stage="fresh_regen",
            output_path=None,
        )

    trace_dir = _resolve_trace_dir(
        arch=arch, slug=slug, shape_point=shape_point, variant_id=variant_id,
    )
    if trace_dir is None:
        return {
            "stage": "fresh_regen",
            "status": "failed_with_typed_reason",
            "typed_failure_reason": "trace_dir_unresolved",
            "output_path": None,
        }
    sidecars_present = _required_sidecars_present(trace_dir)
    if not all(sidecars_present.values()):
        missing = [k for k, v in sidecars_present.items() if not v]
        return {
            "stage": "fresh_regen",
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                "required_sidecars_missing:" + ",".join(sorted(missing))
            ),
            "output_path": None,
            "sidecars_present": sidecars_present,
        }
    mpi_mismatch = _sector_stream_mpi_matches_label(
        trace_dir=trace_dir, label_5tuple=label_5tuple,
    )
    if mpi_mismatch is not None:
        return {
            "stage": "fresh_regen",
            "status": "failed_with_typed_reason",
            "typed_failure_reason": mpi_mismatch,
            "output_path": None,
            "sidecars_present": sidecars_present,
        }
    sector_stream_jsonl_path = (
        trace_dir / "sector_stream" / "sector_stream.jsonl"
    )
    sector_stream_content_reason = _validate_sector_stream_jsonl_records(
        jsonl_path=sector_stream_jsonl_path,
    )
    if sector_stream_content_reason is not None:
        return {
            "stage": "fresh_regen",
            "status": "failed_with_typed_reason",
            "typed_failure_reason": sector_stream_content_reason,
            "output_path": None,
            "sidecars_present": sidecars_present,
        }
    sharing_locality_path = (
        trace_dir / "sharing_locality" / "sharing_locality.json"
    )
    hrd_profile_path = trace_dir / "hrd_profile" / "hrd_profile.json"
    for sidecar_path, schema_path, producer_label in (
        (
            sharing_locality_path,
            _SHARING_LOCALITY_SCHEMA_PATH,
            "sharing_locality",
        ),
        (hrd_profile_path, _HRD_PROFILE_SCHEMA_PATH, "hrd_profile"),
    ):
        schema_reason = _validate_sidecar_against_schema(
            sidecar_path=sidecar_path,
            schema_path=schema_path,
            producer_label=producer_label,
        )
        if schema_reason is not None:
            return {
                "stage": "fresh_regen",
                "status": "failed_with_typed_reason",
                "typed_failure_reason": schema_reason,
                "output_path": None,
                "sidecars_present": sidecars_present,
            }
        mpi_reason = _validate_sidecar_mpi_matches_label(
            sidecar_path=sidecar_path,
            label_5tuple=label_5tuple,
            producer_label=producer_label,
        )
        if mpi_reason is not None:
            return {
                "stage": "fresh_regen",
                "status": "failed_with_typed_reason",
                "typed_failure_reason": mpi_reason,
                "output_path": None,
                "sidecars_present": sidecars_present,
            }
    capacity_bytes = _L2_CAPACITY_BYTES_BY_ARCH.get(arch)
    device_name = _CANONICAL_DEVICE_BY_ARCH.get(arch)
    if capacity_bytes is None or device_name is None:
        return {
            "stage": "fresh_regen",
            "status": "failed_with_typed_reason",
            "typed_failure_reason": f"unknown_arch:{arch}",
            "output_path": None,
        }
    if fresh_regen_run_id is None:
        fresh_regen_run_id = _dt.datetime.now(tz=_dt.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ",
        )
    bundle_stem = (
        f"{arch}__{slug}__{shape_point}" if variant_id is None
        else f"{arch}__{slug}__{shape_point}__{variant_id}"
    )
    output_path = (
        _FRESH_REGEN_ROOT
        / f"sm86_cohort_regen_{fresh_regen_run_id}"
        / "comparison"
        / "predictor_bundles"
        / f"{bundle_stem}.json"
    )
    if dry_run:
        return {
            "stage": "fresh_regen",
            "status": "dry_run_skipped",
            "typed_failure_reason": None,
            "output_path": str(output_path.relative_to(_REPO_ROOT)),
            "trace_dir": str(trace_dir.relative_to(_REPO_ROOT)),
        }
    try:
        from tools.local_labels.local_cache_model_geometry import (  # noqa: E402
            UnsupportedLocalCacheModelGeometryArch,
            resolve_local_cache_model_geometry_for_arch,
        )
        try:
            geometry = resolve_local_cache_model_geometry_for_arch(arch)
        except UnsupportedLocalCacheModelGeometryArch:
            # Fail closed rather than predict under the walker's
            # single-slice / interleave-off fallback for an arch without a
            # populated device-properties manifest key (sm_120 is the
            # sm_120 session's to add). A wrong-geometry bundle would be
            # incomparable to the sidecars / headline runner.
            return {
                "stage": "fresh_regen",
                "status": "failed_with_typed_reason",
                "typed_failure_reason": (
                    f"unsupported_local_cache_model_geometry_arch:{arch}"
                ),
                "output_path": str(output_path.relative_to(_REPO_ROOT)),
            }
        sys.path.insert(0, str(_REPO_ROOT))
        from workloads.simbench_bw.cache_predictor import (  # noqa: E402
            predict_cache_hit_rate,
            write_bundle,
        )
        # Thread the SAME device-properties cache-model geometry the
        # headline runner uses, so a fresh-regen bundle is predicted under
        # the 3090 Ti's real L2-slice / CTA-interleave / HRD-second-layer
        # model (resolved from the manifest) rather than the walker's
        # single-slice / interleave-off / second-layer-off fallback —
        # otherwise this REBUILD_BUNDLE_LOCAL bundle would be computed
        # under a different cache model than the trace sidecars / headline
        # runner and admitted with wrong predictions/provenance.
        bundle = predict_cache_hit_rate(
            trace_dir=trace_dir,
            sector_stream_path=(
                trace_dir / "sector_stream" / "sector_stream.jsonl"
            ),
            cache_model_capacity_bytes=geometry.entry.l2_capacity_bytes,
            manifest_pairing_index=label_5tuple,
            device=device_name,
            sharing_locality_path=(
                trace_dir / "sharing_locality" / "sharing_locality.json"
            ),
            hrd_profile_path=(
                trace_dir / "hrd_profile" / "hrd_profile.json"
            ),
            cache_model_num_l2_slices=geometry.entry.num_l2_slices,
            cache_model_num_l2_slices_source="device_properties_manifest",
            cache_model_cta_interleave_enabled=(
                geometry.cta_interleave_enabled
            ),
            cache_model_hrd_second_layer_enabled=(
                geometry.hrd_second_layer_enabled
            ),
            cache_model_geometry_provenance=geometry.provenance,
        )
        # Stamp the producer's git HEAD as ``predictor_build_commit``
        # so the ledger's admission gate (which requires the
        # producer-bearing tree to be at-or-after the v5 pin) can
        # verify the bundle. The cache_predictor walker is
        # repo-state-agnostic; the producer-provenance pin lives
        # here at the wrapper layer.
        head_sha = _git_head_sha()
        if head_sha is None:
            return {
                "stage": "fresh_regen",
                "status": "failed_with_typed_reason",
                "typed_failure_reason": "git_head_sha_unresolved",
                "output_path": str(
                    output_path.relative_to(_REPO_ROOT),
                ),
            }
        bundle["predictor_build_commit"] = head_sha
        write_bundle(bundle, output_path)
    except Exception as exc:  # noqa: BLE001 — wrap all producer errors typed
        return {
            "stage": "fresh_regen",
            "status": "failed_with_typed_reason",
            "typed_failure_reason": (
                f"walker_raised:{type(exc).__name__}:{exc!s}"
            )[:512],
            "output_path": str(output_path.relative_to(_REPO_ROOT)),
        }
    return {
        "stage": "fresh_regen",
        "status": "written",
        "typed_failure_reason": None,
        "output_path": str(output_path.relative_to(_REPO_ROOT)),
        "predictor_build_commit": head_sha,
    }


def build_predictor_bundle_for_cell(
    *,
    arch: str,
    slug: str,
    shape_point: str,
    variant_id: str | None = None,
    label_admitted_path: Path = _LABEL_ADMITTED,
    fresh_regen_run_id: str | None = None,
    dry_run: bool = False,
    pin_commit: str = _V5_PIN_COMMIT,
) -> dict[str, Any]:
    """Run the two-stage admission cascade for one cell.

    Returns a per-cell report dict carrying the ledger-shaped
    fields ``bundle_admission_path`` (``legacy_match`` /
    ``fresh_regen`` / ``unavailable``), ``status``,
    ``typed_failure_reason``, ``output_path``, and per-stage
    diagnostics.

    ``variant_id`` selects the build-variant sibling: the right
    per-cap ``binary_sha256`` is resolved for Stage-1 identity
    matching, and Stage-2 fresh_regen writes a distinct per-cap
    bundle file.
    """
    cell_key = (
        f"{arch}/{slug}/{shape_point}" if variant_id is None
        else f"{arch}/{slug}/{shape_point}/{variant_id}"
    )
    if is_sm120_template_arch(arch):
        return unavailable_result(
            operation="build_predictor_bundle_for_cell",
            arch=arch,
            cell_key=cell_key,
            bundle_admission_path="unavailable",
            stage_1=None,
            stage_2=None,
            output_path=None,
        )

    label_5tuple = _label_5tuple_for_cell(
        arch=arch,
        slug=slug,
        shape_point=shape_point,
        variant_id=variant_id,
        label_admitted_path=label_admitted_path,
    )
    if label_5tuple is None:
        return {
            "cell_key": cell_key,
            "bundle_admission_path": None,
            "status": "failed_with_typed_reason",
            "typed_failure_reason": "label_5tuple_unresolved",
            "stage_1": None,
            "stage_2": None,
        }
    stage_1 = legacy_match_scan(
        arch=arch, slug=slug, shape_point=shape_point,
        label_5tuple=label_5tuple, pin_commit=pin_commit,
    )
    if stage_1["admitted"]:
        return {
            "cell_key": cell_key,
            "bundle_admission_path": "legacy_match",
            "status": "admitted",
            "typed_failure_reason": None,
            "stage_1": stage_1,
            "stage_2": None,
            "output_path": stage_1["path"],
        }
    stage_2 = fresh_regen_via_walker(
        arch=arch, slug=slug, shape_point=shape_point,
        label_5tuple=label_5tuple, variant_id=variant_id,
        fresh_regen_run_id=fresh_regen_run_id,
        dry_run=dry_run,
    )
    if stage_2["status"] == "written":
        return {
            "cell_key": cell_key,
            "bundle_admission_path": "fresh_regen",
            "status": "admitted",
            "typed_failure_reason": None,
            "stage_1": stage_1,
            "stage_2": stage_2,
            "output_path": stage_2["output_path"],
        }
    return {
        "cell_key": cell_key,
        "bundle_admission_path": "unavailable",
        "status": stage_2["status"],
        "typed_failure_reason": stage_2.get("typed_failure_reason"),
        "stage_1": stage_1,
        "stage_2": stage_2,
        "output_path": stage_2.get("output_path"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug", required=True,
        help="Slug name (e.g., bf16_hmma_gemm_traceable).",
    )
    parser.add_argument(
        "--shape-point", required=True,
        help="Shape point (e.g., trace_l2_accuracy_xl, timing_shape).",
    )
    parser.add_argument(
        "--arch", required=True,
        help="Cell architecture (sm_86 or sm_120).",
    )
    parser.add_argument(
        "--variant-id", default=None,
        help=(
            "Optional build-variant id (e.g. register_cap_64). When "
            "set, resolves the per-cap variant trace + label row and "
            "writes a distinct per-cap bundle filename."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_LABEL_ADMITTED,
        help=(
            "Label-admitted JSON used to resolve the cell identity "
            "5-tuple (default: data/local_labels/label_admitted.json)."
        ),
    )
    parser.add_argument(
        "--fresh-regen-run-id", default=None,
        help=(
            "Optional UTC timestamp for the fresh_regen output root "
            "(defaults to UTC now)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Run Stage 1 and resolve Stage 2 prerequisites WITHOUT "
            "invoking the walker. Emits a structured report."
        ),
    )
    args = parser.parse_args(argv)
    report = build_predictor_bundle_for_cell(
        arch=args.arch,
        slug=args.slug,
        shape_point=args.shape_point,
        variant_id=args.variant_id,
        label_admitted_path=args.label_admitted,
        fresh_regen_run_id=args.fresh_regen_run_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    # A dry-run remains non-failing for the active sm_86 cascade, but the
    # sm_120 template gate is an intentional unavailable result rather than a
    # successful no-op. Do not let --dry-run mask that safety decision.
    if report.get("status") == "typed_unavailable":
        return 2
    if args.dry_run:
        return 0
    return 0 if report.get("status") == "admitted" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
