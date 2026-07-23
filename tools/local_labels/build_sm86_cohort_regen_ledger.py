"""Per-cell regeneration ledger producer for the sm_86 cohort.

The ledger lives at
``data/local_labels/sm86_cohort_regen_ledger.json`` and covers every
sm_86 cell currently in ``data/local_labels/label_admitted.json``.
Each entry carries the 6-tuple identity key
``(arch, slug, shape_point, binary_sha256,
kernel_signature_sha256, label_source_commit)`` plus the closed
terminal status, regen-artifact paths, and per-artifact SHA-256
digests for admitted cells.

The status enum is closed:
``{admitted, rebuild_binary_failed, trace_failed, sidecar_failed,
bundle_failed, schema_invalid, identity_mismatch,
trace_volume_exceeded, slowdown_budget_exceeded,
drop_rate_exceeded}``. A status outside this enum is rejected by
``workloads/simbench_bw/schemas/sm86_cohort_regen_ledger.schema.json``.

This producer evaluates each cell's regen state by inspecting the
on-disk inventory:

* binary at ``workloads/<workload>/<slug>/<slug>_run`` or
  ``workloads/simbench_bw/l1/<slug>/<slug>``;
* trace directory under ``workloads/simbench_bw/l1/<slug>/traces/...``;
* predictor bundle under ``data/local_labels/sweeps/.../predictor_bundles/``;
* sidecars under the trace directory (sharing_locality, hrd_profile,
  sector_stream).

When the regen pipeline (task3 trace + task4 sidecar + task5 bundle
cascade) has not been run against the refreshed labels, the producer
emits ``status="trace_failed"`` with
``typed_reason="dynamic_trace_pb_absent_post_relabel"`` to honestly
record the pre-cascade state. The ledger transitions to
``status="admitted"`` once the cascade artifacts land for the cell.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Reuse the per-cell bundle builder's already-hardened validation
# helpers so the ledger and the bundle producer share a single
# admission contract. The import path resolves whether this module
# is invoked as a script (``python tools/local_labels/<name>.py``)
# or imported as a package member.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from build_predictor_bundle_for_cell import (  # noqa: E402
    _DIVERGENCE_PROXIES_SCHEMA_PATH,
    _HRD_PROFILE_SCHEMA_PATH,
    _ORDER_PROJECTIONS_SCHEMA_PATH,
    _SHARING_LOCALITY_SCHEMA_PATH,
    _V5_PIN_COMMIT,
    _evaluate_candidate,
    _sector_stream_mpi_matches_label,
    _validate_sector_stream_jsonl_records,
    _validate_sidecar_against_schema,
    _validate_sidecar_mpi_matches_label,
)
# Shared mixed-arity build-variant identity helper (sibling import; the SAME
# normalizer the corpus / selector / merge / sidecar-regenerator use). A base
# cell yields "" and a register-cap variant yields its canonical
# ``register_cap_<n>`` (raising on an inconsistent pair).
from corpus_cell_identity import cell_variant_id  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABEL_ADMITTED = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
_LEDGER_OUT_DEFAULT = (
    _REPO_ROOT / "data" / "local_labels" / "sm86_cohort_regen_ledger.json"
)

# Workload classification: which workload tree owns the per-slug binary?
# simbench_bw cells have binaries at workloads/simbench_bw/l1/<slug>/<slug>[_run]
# or workloads/simbench_bw/l2/<slug>/<slug>[_run] (the dual-workload l2 tree,
# e.g. rmsnorm_fwd_bf16); portable_lcd cells have binaries at
# workloads/portable_lcd/<slug>/<slug>_run.
_SIMBENCH_L1_DIR = _REPO_ROOT / "workloads" / "simbench_bw" / "l1"
_SIMBENCH_L2_DIR = _REPO_ROOT / "workloads" / "simbench_bw" / "l2"
_PORTABLE_LCD_DIR = _REPO_ROOT / "workloads" / "portable_lcd"
_HOST_SKU = "rtx_3090_ti"
_HOST_ARCH = "sm_86"


def _classify_workload(slug: str) -> str | None:
    """Return 'simbench_bw' or 'portable_lcd' based on which workload
    tree owns this slug's source directory; None when neither tree
    has a directory for the slug (a contract drift)."""
    simbench = _SIMBENCH_L1_DIR / slug
    simbench_l2 = _SIMBENCH_L2_DIR / slug
    portable = _PORTABLE_LCD_DIR / slug
    sb = simbench.is_dir() or simbench_l2.is_dir()
    pl = portable.is_dir()
    if sb and not pl:
        return "simbench_bw"
    if pl and not sb:
        return "portable_lcd"
    return None


def _git_head_sha() -> str:
    """Return the current HEAD commit SHA-1 (40 hex chars)."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_binary_path(
    slug: str, *, register_cap: int | None = None,
) -> Path | None:
    """Return the repo-relative path to the per-slug binary if it
    exists on disk; None otherwise. Checks both workload trees and
    both naming conventions (with/without ``_run`` suffix).

    ``register_cap`` is additive: when set, the per-cap build-variant
    binary at ``<slug>/spill_variants/<slug>_run__cap<N>`` is resolved
    (the convention the register-cap variant producer writes), so a
    variant ledger entry records the DISTINCT per-cap binary rather
    than the (absent) shared base binary."""
    if register_cap is not None:
        variant_binary = (
            _SIMBENCH_L1_DIR / slug / "spill_variants"
            / f"{slug}_run__cap{register_cap}"
        )
        return variant_binary if variant_binary.is_file() else None
    candidates = (
        _SIMBENCH_L1_DIR / slug / slug,
        _SIMBENCH_L1_DIR / slug / f"{slug}_run",
        _SIMBENCH_L2_DIR / slug / f"{slug}_run",
        _SIMBENCH_L2_DIR / slug / slug,
        _PORTABLE_LCD_DIR / slug / f"{slug}_run",
        _PORTABLE_LCD_DIR / slug / slug,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_trace_dir(
    slug: str, shape_point: str, *, variant_id: str | None = None,
) -> Path | None:
    """Return the trace directory for the cell if it contains
    ``dynamic_trace.pb``; None otherwise. Uses the canonical
    convention shared with ``emit_sm86_cohort_preflight.py`` and
    ``build_predictor_bundle_for_cell.py``:
    ``workloads/<suite>/results/<slug>/rtx_3090_ti/sm_86/trace_<shape_point>/``.

    ``variant_id`` is additive: when set, the per-cap variant subtree
    ``.../sm_86/variants/<variant_id>/trace_<shape_point>/`` is
    resolved instead of the base trace dir, so sibling caps never
    collapse onto one trace directory."""
    workload = _classify_workload(slug)
    if workload is None:
        return None
    leaf = (
        Path(f"trace_{shape_point}") if variant_id is None
        else Path("variants") / variant_id / f"trace_{shape_point}"
    )
    candidate = (
        _REPO_ROOT
        / "workloads"
        / workload
        / "results"
        / slug
        / _HOST_SKU
        / _HOST_ARCH
        / leaf
    )
    if candidate.is_dir() and (candidate / "dynamic_trace.pb").is_file():
        return candidate
    return None


def _resolve_sidecar_paths(
    trace_dir: Path | None,
) -> dict[str, Path | None]:
    """Given a trace_dir, return the canonical sidecar paths if they
    exist. The five primary sidecars are produced under the trace
    directory in their respective subfolders; the
    ``sector_stream/manifest_pairing_index.json`` sibling pins the
    bundle-identity binding for the JSONL.

    The five primary sidecars match the original task4 producer
    inventory (`plans/sm86_cohort_regen_3090ti_plan.md:122` +
    `:176`):
    sharing_locality, hrd_profile, sector_stream,
    divergence_proxies, and order_projections. A cell missing
    any of the five (or the sector_stream MPI sibling) routes to
    ``status=sidecar_failed`` via the validator."""
    if trace_dir is None:
        return {
            "sharing_locality": None,
            "hrd_profile": None,
            "sector_stream": None,
            "sector_stream_mpi": None,
            "divergence_proxies": None,
            "order_projections": None,
        }
    sharing = trace_dir / "sharing_locality" / "sharing_locality.json"
    hrd = trace_dir / "hrd_profile" / "hrd_profile.json"
    sector = trace_dir / "sector_stream" / "sector_stream.jsonl"
    sector_mpi = (
        trace_dir / "sector_stream" / "manifest_pairing_index.json"
    )
    divergence = (
        trace_dir / "divergence_proxies" / "divergence_proxies.json"
    )
    order = (
        trace_dir / "order_projections" / "order_projections.json"
    )
    return {
        "sharing_locality": sharing if sharing.is_file() else None,
        "hrd_profile": hrd if hrd.is_file() else None,
        "sector_stream": sector if sector.is_file() else None,
        "sector_stream_mpi": sector_mpi if sector_mpi.is_file() else None,
        "divergence_proxies": divergence if divergence.is_file() else None,
        "order_projections": order if order.is_file() else None,
    }


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify_admission_path(bundle_path: Path) -> str:
    """Classify a bundle path as ``legacy_match`` or
    ``fresh_regen``. Fresh-regen bundles live under
    ``data/local_labels/sweeps/sm86_cohort_regen_<UTC>/comparison/predictor_bundles/``
    (the convention pinned by
    ``build_predictor_bundle_for_cell.fresh_regen_via_walker``).
    Every other matching path is legacy_match (the prior
    cross-arch sweep roots like
    ``sweep_<timestamp>_<hash>/predictor_bundles/``)."""
    try:
        rel = bundle_path.relative_to(_REPO_ROOT)
    except ValueError:
        rel = bundle_path
    parts = rel.parts
    for part in parts:
        if part.startswith("sm86_cohort_regen_"):
            return "fresh_regen"
    return "legacy_match"


def _resolve_predictor_bundle(
    *,
    slug: str,
    shape_point: str,
    binary_sha256: str,
    kernel_signature_sha256: str,
    label_source_commit: str,
) -> dict[str, Any]:
    """Walk every bundle under
    ``data/local_labels/sweeps/**/predictor_bundles/*.json`` whose
    ``manifest_pairing_index`` matches the cell's full 6-tuple
    identity (arch=sm_86, slug, shape_point, binary_sha256,
    kernel_signature_sha256, source_commit=label_source_commit).
    Evaluate every match via ``_evaluate_candidate`` and admit the
    first one that passes every gate.

    Returns a structured verdict:

    * ``found``: bool — at least one identity-matching bundle.
    * ``admitted``: bool — at least one matching candidate passed
      every gate.
    * ``path``: Path | None — repo path of the admitted bundle.
    * ``admission_path``: str | None — ``legacy_match`` /
      ``fresh_regen`` for admitted; None otherwise.
    * ``predictor_build_commit``: str | None — the admitted
      candidate's producer commit.
    * ``typed_failure_reason``: str | None — populated when matches
      exist but none admit (``all_bundle_candidates_rejected:<n>:<first_reason>``)
      or when no identity match exists
      (``predictor_bundle_absent_for_refreshed_identity``).
    * ``rejected_candidates``: list[dict] — per-candidate rejection
      diagnostics in scan order (path + typed_failure_reason +
      predictor_build_commit).

    Mirrors ``build_predictor_bundle_for_cell.legacy_match_scan``'s
    discipline so the ledger and the per-cell builder share a
    single candidate-cascade semantics."""
    sweeps_root = _REPO_ROOT / "data" / "local_labels" / "sweeps"
    if not sweeps_root.is_dir():
        return {
            "found": False,
            "admitted": False,
            "path": None,
            "admission_path": None,
            "predictor_build_commit": None,
            "typed_failure_reason": (
                "predictor_bundle_absent_for_refreshed_identity"
            ),
            "rejected_candidates": [],
        }
    identity_matches: list[tuple[Path, dict]] = []
    for bundle in sorted(
        sweeps_root.rglob("predictor_bundles/*.json"),
    ):
        try:
            payload = json.loads(bundle.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mpi = payload.get("manifest_pairing_index", {})
        if not isinstance(mpi, dict):
            continue
        # The MPI 5-tuple is closed to {slug, shape_point,
        # binary_sha256, source_commit, kernel_signature_sha256}
        # per the sharing_locality / hrd_profile / cache_predictor
        # schemas' `additionalProperties: false`. `arch` is
        # encoded in the file path (e.g.,
        # `sm_86__<slug>__<shape>.json`) and the trace_dir tree,
        # NOT in the MPI block. Filter by the 5-tuple ONLY; the
        # arch routing is implicit in the cohort iteration (this
        # ledger only iterates sm_86 cells from
        # label_admitted.json).
        if (
            mpi.get("slug") == slug
            and mpi.get("shape_point") == shape_point
            and mpi.get("binary_sha256") == binary_sha256
            and mpi.get("kernel_signature_sha256")
                == kernel_signature_sha256
            and mpi.get("source_commit") == label_source_commit
        ):
            identity_matches.append((bundle, payload))
    if not identity_matches:
        return {
            "found": False,
            "admitted": False,
            "path": None,
            "admission_path": None,
            "predictor_build_commit": None,
            "typed_failure_reason": (
                "predictor_bundle_absent_for_refreshed_identity"
            ),
            "rejected_candidates": [],
        }
    rejected: list[dict[str, Any]] = []
    for bundle, payload in identity_matches:
        verdict = _evaluate_candidate(
            payload=payload, pin_commit=_V5_PIN_COMMIT,
        )
        if verdict["admitted"]:
            return {
                "found": True,
                "admitted": True,
                "path": bundle,
                "admission_path": _classify_admission_path(bundle),
                "predictor_build_commit": verdict[
                    "predictor_build_commit"
                ],
                "typed_failure_reason": None,
                "rejected_candidates": rejected,
            }
        try:
            rel_str = str(bundle.relative_to(_REPO_ROOT))
        except ValueError:
            rel_str = str(bundle)
        rejected.append({
            "path": rel_str,
            "typed_failure_reason": verdict["typed_failure_reason"],
            "predictor_build_commit": verdict[
                "predictor_build_commit"
            ],
        })
    first_reason = (
        rejected[0]["typed_failure_reason"] if rejected else "unknown"
    )
    return {
        "found": True,
        "admitted": False,
        "path": None,
        "admission_path": None,
        "predictor_build_commit": None,
        "typed_failure_reason": (
            f"all_bundle_candidates_rejected:"
            f"{len(rejected)}:{first_reason}"
        ),
        "rejected_candidates": rejected,
    }


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _validate_sidecars_for_cell(
    *,
    trace_dir: Path,
    sidecars: dict[str, Path | None],
    label_5tuple: dict[str, str],
) -> str | None:
    """Run the sidecar half of the admission cascade (schema +
    MPI identity for sharing_locality / hrd_profile +
    sector_stream JSONL per-record schema + sector_stream MPI
    sibling identity) and return the first typed reject reason,
    or None when every sidecar validates.

    This helper exists so the ledger's `_evaluate_cell` can
    enforce sidecar correctness BEFORE checking predictor-bundle
    absence. Without it, a cell with invalid sidecars but no
    matching bundle would mask the sidecar failure as
    `bundle_failed`, hiding a real sidecar-identity-binding
    violation.
    """
    sharing = sidecars["sharing_locality"]
    hrd = sidecars["hrd_profile"]
    sector = sidecars["sector_stream"]

    reason = _validate_sidecar_against_schema(
        sidecar_path=sharing,
        schema_path=_SHARING_LOCALITY_SCHEMA_PATH,
        producer_label="sharing_locality",
    )
    if reason is not None:
        return reason
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=sharing,
        label_5tuple=label_5tuple,
        producer_label="sharing_locality",
    )
    if reason is not None:
        return reason
    reason = _validate_sidecar_against_schema(
        sidecar_path=hrd,
        schema_path=_HRD_PROFILE_SCHEMA_PATH,
        producer_label="hrd_profile",
    )
    if reason is not None:
        return reason
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=hrd,
        label_5tuple=label_5tuple,
        producer_label="hrd_profile",
    )
    if reason is not None:
        return reason
    reason = _validate_sector_stream_jsonl_records(jsonl_path=sector)
    if reason is not None:
        return reason
    reason = _sector_stream_mpi_matches_label(
        trace_dir=trace_dir, label_5tuple=label_5tuple,
    )
    if reason is not None:
        return reason
    # divergence_proxies: schema + MPI identity.
    divergence = sidecars["divergence_proxies"]
    reason = _validate_sidecar_against_schema(
        sidecar_path=divergence,
        schema_path=_DIVERGENCE_PROXIES_SCHEMA_PATH,
        producer_label="divergence_proxies",
    )
    if reason is not None:
        return reason
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=divergence,
        label_5tuple=label_5tuple,
        producer_label="divergence_proxies",
    )
    if reason is not None:
        return reason
    # order_projections: schema + MPI identity.
    order = sidecars["order_projections"]
    reason = _validate_sidecar_against_schema(
        sidecar_path=order,
        schema_path=_ORDER_PROJECTIONS_SCHEMA_PATH,
        producer_label="order_projections",
    )
    if reason is not None:
        return reason
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=order,
        label_5tuple=label_5tuple,
        producer_label="order_projections",
    )
    if reason is not None:
        return reason
    return None


def _admission_gate_cascade(
    *,
    trace_dir: Path,
    sidecars: dict[str, Path | None],
    bundle_path: Path,
    label_5tuple: dict[str, str],
) -> tuple[str, str | None, str]:
    """Run the full validation cascade against the resolved
    artifacts. Returns ``(status, typed_reason, validator_status)``.

    The cascade order mirrors the per-cell bundle builder's
    fresh-regen admission path so the ledger and the builder share
    a single validation contract:

    1. sharing_locality schema + MPI identity
    2. hrd_profile schema + MPI identity
    3. sector_stream JSONL per-record schema + MPI sibling identity
    4. predictor bundle schema + canonical 6-feature block +
       forbidden digest keys + predictor_build_commit ancestry

    Any failed gate routes to the matching typed-failure status with
    the gate's typed reason carried verbatim. Only an end-to-end
    pass returns ``("admitted", None, "passed")``.
    """
    sharing = sidecars["sharing_locality"]
    hrd = sidecars["hrd_profile"]
    sector = sidecars["sector_stream"]

    # sharing_locality: schema + MPI identity.
    reason = _validate_sidecar_against_schema(
        sidecar_path=sharing,
        schema_path=_SHARING_LOCALITY_SCHEMA_PATH,
        producer_label="sharing_locality",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=sharing,
        label_5tuple=label_5tuple,
        producer_label="sharing_locality",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"

    # hrd_profile: schema + MPI identity.
    reason = _validate_sidecar_against_schema(
        sidecar_path=hrd,
        schema_path=_HRD_PROFILE_SCHEMA_PATH,
        producer_label="hrd_profile",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=hrd,
        label_5tuple=label_5tuple,
        producer_label="hrd_profile",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"

    # sector_stream: per-record schema + MPI sibling identity.
    reason = _validate_sector_stream_jsonl_records(jsonl_path=sector)
    if reason is not None:
        return "sidecar_failed", reason, "failed"
    reason = _sector_stream_mpi_matches_label(
        trace_dir=trace_dir, label_5tuple=label_5tuple,
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"

    # divergence_proxies: schema + MPI identity.
    divergence = sidecars["divergence_proxies"]
    reason = _validate_sidecar_against_schema(
        sidecar_path=divergence,
        schema_path=_DIVERGENCE_PROXIES_SCHEMA_PATH,
        producer_label="divergence_proxies",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=divergence,
        label_5tuple=label_5tuple,
        producer_label="divergence_proxies",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"

    # order_projections: schema + MPI identity.
    order = sidecars["order_projections"]
    reason = _validate_sidecar_against_schema(
        sidecar_path=order,
        schema_path=_ORDER_PROJECTIONS_SCHEMA_PATH,
        producer_label="order_projections",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"
    reason = _validate_sidecar_mpi_matches_label(
        sidecar_path=order,
        label_5tuple=label_5tuple,
        producer_label="order_projections",
    )
    if reason is not None:
        return "sidecar_failed", reason, "failed"

    # Predictor bundle: schema + canonical features + forbidden
    # digest keys + predictor_build_commit ancestry.
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (
            "bundle_failed",
            f"predictor_bundle_unreadable:"
            f"{type(exc).__name__}:{exc!s}"[:256],
            "failed",
        )
    verdict = _evaluate_candidate(
        payload=payload, pin_commit=_V5_PIN_COMMIT,
    )
    if not verdict["admitted"]:
        return (
            "bundle_failed",
            verdict["typed_failure_reason"],
            "failed",
        )
    return "admitted", None, "passed"


def _evaluate_cell(
    cell: dict[str, Any],
    *,
    attempted_at: str,
    producer_commit: str,
) -> dict[str, Any]:
    """Build one ledger entry for a sm_86 cell. The status reflects
    the current on-disk inventory: if every regen artifact is
    present + schema-valid, status=admitted; otherwise the producer
    maps to the closest closed-enum status with a typed_reason."""
    slug = cell["slug"]
    shape_point = cell["shape_point"]
    binary_sha256 = cell["binary_sha256"]
    kernel_signature_sha256 = cell["kernel_signature_sha256"]
    label_source_commit = cell["source_commit"]
    # Mixed-arity build-variant identity (canonical "" for base cells,
    # register_cap_<n> for variants; raises on an inconsistent pair).
    variant_id = cell_variant_id(cell) or None
    register_cap = cell.get("register_cap")

    binary_path = _resolve_binary_path(slug, register_cap=register_cap)
    trace_dir = _resolve_trace_dir(slug, shape_point, variant_id=variant_id)
    sidecars = _resolve_sidecar_paths(trace_dir)
    bundle_verdict = _resolve_predictor_bundle(
        slug=slug,
        shape_point=shape_point,
        binary_sha256=binary_sha256,
        kernel_signature_sha256=kernel_signature_sha256,
        label_source_commit=label_source_commit,
    )
    bundle_path: Path | None = bundle_verdict["path"]
    bundle_admission_path: str | None = None

    # Status discovery: walk the regen pipeline in order. When every
    # artifact resolves and identity-binds to the label, the cell is
    # admitted; otherwise the first absent artifact pins the status.
    status: str
    typed_reason: str | None
    validator_status: str
    if binary_path is None:
        status = "rebuild_binary_failed"
        typed_reason = (
            f"per_slug_binary_absent_at:"
            f"workloads/<workload>/{slug}"
        )
        validator_status = "not_run"
    elif trace_dir is None:
        status = "trace_failed"
        typed_reason = "dynamic_trace_pb_absent_post_relabel"
        validator_status = "not_run"
    elif any(p is None for p in sidecars.values()):
        missing = [k for k, v in sidecars.items() if v is None]
        status = "sidecar_failed"
        typed_reason = (
            f"sidecar_absent_post_relabel:{','.join(missing)}"
        )
        validator_status = "not_run"
    elif (
        sidecar_reason := _validate_sidecars_for_cell(
            trace_dir=trace_dir,
            sidecars=sidecars,
            label_5tuple={
                "slug": slug,
                "shape_point": shape_point,
                "binary_sha256": binary_sha256,
                "source_commit": label_source_commit,
                "kernel_signature_sha256": kernel_signature_sha256,
            },
        )
    ) is not None:
        # Sidecars are present on disk but fail schema or MPI
        # identity validation. This MUST NOT silently fall through
        # to `bundle_failed`; the cell stays at `sidecar_failed`
        # with the validator's typed reason. Predictor bundle
        # absence is reported only after the sidecar contract holds.
        status = "sidecar_failed"
        typed_reason = sidecar_reason
        validator_status = "failed"
    elif not bundle_verdict["admitted"]:
        # Either no identity match exists (resolver returned the
        # no-match verdict), or matches exist but none passed every
        # bundle gate (the all-candidates-rejected verdict). The
        # resolver's typed_failure_reason already names the
        # aggregate; carry it verbatim into the ledger.
        status = "bundle_failed"
        typed_reason = bundle_verdict["typed_failure_reason"]
        validator_status = "not_run" if not bundle_verdict[
            "found"
        ] else "failed"
    else:
        # The resolver admitted one candidate (the first valid
        # match in scan order). Run the SAME sidecar + bundle
        # validation gates the per-cell bundle builder applies for
        # the fresh-regen admission path. The bundle gate inside
        # the cascade re-evaluates the admitted candidate as a
        # belt-and-suspenders check; the resolver already passed
        # every gate, but running the cascade against the same
        # payload is cheap and pins the wire-boundary contract.
        label_5tuple = {
            "slug": slug,
            "shape_point": shape_point,
            "binary_sha256": binary_sha256,
            "source_commit": label_source_commit,
            "kernel_signature_sha256": kernel_signature_sha256,
        }
        status, typed_reason, validator_status = (
            _admission_gate_cascade(
                trace_dir=trace_dir,
                sidecars=sidecars,
                bundle_path=bundle_path,
                label_5tuple=label_5tuple,
            )
        )
        if status == "admitted":
            bundle_admission_path = bundle_verdict["admission_path"]

    # Build the entry. The artifact fields are populated even for
    # non-admitted cells if the artifacts happen to exist on disk;
    # this gives reviewers the inventory state without lying about
    # admission.
    sidecar_paths_out: dict[str, str | None] = {
        k: _rel(v) for k, v in sidecars.items()
    }
    if all(v is None for v in sidecar_paths_out.values()):
        sidecar_paths_field: dict[str, str | None] | None = None
    else:
        sidecar_paths_field = sidecar_paths_out

    artifact_sha256: dict[str, str | None] = {
        "predictor_bundle": _file_sha256(bundle_path),
        "dynamic_trace_pb": _file_sha256(
            trace_dir / "dynamic_trace.pb" if trace_dir else None,
        ),
        "sharing_locality": _file_sha256(sidecars["sharing_locality"]),
        "hrd_profile": _file_sha256(sidecars["hrd_profile"]),
        "sector_stream": _file_sha256(sidecars["sector_stream"]),
        "sector_stream_mpi": _file_sha256(
            sidecars["sector_stream_mpi"],
        ),
        "divergence_proxies": _file_sha256(
            sidecars["divergence_proxies"],
        ),
        "order_projections": _file_sha256(
            sidecars["order_projections"],
        ),
    }
    if all(v is None for v in artifact_sha256.values()):
        artifact_sha256_field: dict[str, str | None] | None = None
    else:
        artifact_sha256_field = artifact_sha256

    entry: dict[str, Any] = {
        "arch": "sm_86",
        "slug": slug,
        "shape_point": shape_point,
        "binary_sha256": binary_sha256,
        "kernel_signature_sha256": kernel_signature_sha256,
        "label_source_commit": label_source_commit,
        "status": status,
        "typed_reason": typed_reason,
        "binary_path": _rel(binary_path),
        "trace_dir": _rel(trace_dir),
        "predictor_bundle_path": _rel(bundle_path),
        "bundle_admission_path": bundle_admission_path,
        "sidecar_paths": sidecar_paths_field,
        "attempted_at": attempted_at,
        "producer_commit": producer_commit,
        "artifact_sha256": artifact_sha256_field,
        "validator_status": validator_status,
        "live_evidence_status": "Verified-live-sm_86",
    }
    # Additive build-variant identity: base cells omit both fields and stay
    # byte-identical; register-cap variants carry the canonical pair so the
    # coverage scanner can mixed-arity-key them apart from base/siblings.
    if variant_id is not None:
        entry["variant_id"] = variant_id
        if register_cap is not None:
            entry["register_cap"] = register_cap
    return entry


def build_ledger(
    *,
    label_admitted_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Read label_admitted.json, evaluate every sm_86 cell, emit the
    ledger artifact to ``out_path`` and return the artifact dict."""
    payload = json.loads(label_admitted_path.read_text(encoding="utf-8"))
    # Sort key includes the canonical build-variant id so sibling register-cap
    # variants of one (slug, shape_point) order deterministically and never
    # collapse onto each other.
    sm86_cells = sorted(
        [c for c in payload.get("cells", []) if c.get("arch") == "sm_86"],
        key=lambda c: (c["slug"], c["shape_point"], cell_variant_id(c)),
    )
    attempted_at = (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    producer_commit = _git_head_sha()
    ledger = {
        "schema_version": 1,
        "generated_at_utc": attempted_at,
        "source_commit": producer_commit,
        "host_arch": "sm_86",
        "n_cells": len(sm86_cells),
        "cells": [
            _evaluate_cell(
                c,
                attempted_at=attempted_at,
                producer_commit=producer_commit,
            )
            for c in sm86_cells
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the sm_86 cohort regeneration ledger by inspecting "
            "the on-disk regen-artifact inventory."
        ),
    )
    parser.add_argument(
        "--label-admitted",
        type=Path,
        default=_LABEL_ADMITTED,
        help="Path to the label_admitted.json source artifact.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_LEDGER_OUT_DEFAULT,
        help="Destination path for the emitted ledger artifact.",
    )
    args = parser.parse_args(argv)
    ledger = build_ledger(
        label_admitted_path=args.label_admitted,
        out_path=args.out,
    )
    status_counts: dict[str, int] = {}
    for cell in ledger["cells"]:
        status_counts[cell["status"]] = (
            status_counts.get(cell["status"], 0) + 1
        )
    print(
        f"build_sm86_cohort_regen_ledger: wrote "
        f"{args.out.relative_to(_REPO_ROOT)} "
        f"(n_cells={ledger['n_cells']}, status_counts={status_counts})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
