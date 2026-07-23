"""Single-command CLI that runs the full v5 artifact regeneration
sequence + lockstep drift-surface restamp + emits a structured
per-step status report.

The v5 regen sequence (executed manually each round 20+) is a chain
of 6 emitter modules followed by 3 drift-surface restamp steps:

  1. fit_monotonic_calibration_head_v5 — ALWAYS invoked split-aware
     (ledger-driven admitted bundles + the sealed cohort split:
     calibration+validation fit, holdout excluded) targeting the
     standalone sm_86 head; a split-unaware invocation fails closed
     with FitterSplitConfigError BEFORE the fitter runs, and an
     emitted head whose fit_scope does not record the split-aware
     calibration+validation fit fails the step and halts the chain.
  2. nugteren2014_rdh_baseline (reads the sm_86 v5 head)
  3. emit_v5_closure_artifacts (3 closure artifacts, sm_86 head)
  4. emit_v5_robustness_artifact (sm_86 head)
  5. run_v5_active_path_comparison — emits the committed comparison
     bundle (explicit --allow-committed-paths opt-in) while its
     per-arch head emissions stay on the runner's non-committed
     rehydration-scratch defaults, so the canonical fitted heads can
     never be clobbered by this chain (the historical producer
     collision wrote an admitted_count=0 rehydration stub over the
     fitted sm_86 head).
  6. emit_v5_cohort_staging_manifest
  7. refresh_admitted_label_source_commit (per drift-managed label file)
  8. refresh_sweep_build_commit (per drift-managed sweep dir)
  9. variance_trajectory per-cell entry restamp

Steps 7-9 discover their work dynamically:

  - Cells with ``source_commit_refresh_history`` in the label file
    are the drift-managed cell set (step 7).
  - Sweep dirs whose ``regeneration_delta.json::regeneration_build_commit``
    is outside the {HEAD, HEAD^} accept window are the drift-managed
    sweep set (step 8).
  - Per-cell entries in ``variance_trajectory.json::current_per_cell_predictions``
    whose ``predictor_build_commit`` does not equal HEAD are restamped
    (step 9).

Emits ``data/local_labels/v5_artifact_regeneration_report.json`` with
shape::

    {
      "head_at_wrapper_invocation": "<sha>",
      "started_at": "<iso-timestamp>",
      "finished_at": "<iso-timestamp>",
      "steps": {
        "<step_name>": {"status": "ok"|"failed"|"no_action",
                        "exit_code"?: int, "reason"?: str, ...},
        ...
      },
      "overall_status": "ok"|"failed",
    }

CLI::

    python tools/local_labels/regenerate_v5_artifacts.py [--dry-run]

Returns non-zero exit if any step fails (the report still lands so a
reviewer can see which step failed and why).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_DEFAULT_REPORT = (
    REPO_ROOT / "data" / "local_labels"
    / "v5_artifact_regeneration_report.json"
)

# Label files whose drift-managed cells need refresh after each round.
# The set is small + stable; dynamic discovery of label files would be
# wider scope than the current loop needs.
_LABEL_FILES: tuple[Path, ...] = (
    REPO_ROOT / "data" / "local_labels" / "label_admitted.json",
    REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "round_12_strengthening" / "selected_label_admitted.json",
)

# Variance-trajectory artifact whose per-cell predictor_build_commit
# entries are restamped in place when stale.
_VARIANCE_TRAJECTORY = (
    REPO_ROOT / "data" / "local_labels" / "audit_evidence"
    / "ac5_variance_precondition_diagnostic" / "variance_trajectory.json"
)

# Sweeps root for drift-managed sweep dir discovery.
_SWEEPS_ROOT = REPO_ROOT / "data" / "local_labels" / "sweeps"

# Field name on regeneration_delta + predictor bundles.
_REGENERATION_DELTA_FIELD = "regeneration_build_commit"
_PREDICTOR_BUNDLE_FIELD = "predictor_build_commit"


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_parent_sha(head_sha: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"{head_sha}^"], cwd=str(REPO_ROOT),
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def discover_drift_managed_cells(label_path: Path) -> list[str]:
    """Read a label-admitted JSON and return the
    ``arch/slug/shape_point`` IDs for cells carrying
    ``source_commit_refresh_history`` (i.e., cells that have been
    drift-managed at least once and thus need refresh on each
    HEAD-advancing round).

    Returns an empty list when the file is absent, malformed, or
    contains no drift-managed cells.
    """
    if not label_path.is_file():
        return []
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cells = payload.get("cells", []) or []
    out: list[str] = []
    for c in cells:
        if not isinstance(c, dict):
            continue
        if c.get("source_commit_refresh_history") is None:
            continue
        arch = c.get("arch", "")
        slug = c.get("slug", "")
        shape_point = c.get("shape_point", "")
        if arch and slug and shape_point:
            out.append(f"{arch}/{slug}/{shape_point}")
    return out


def discover_drift_managed_sweep_dirs(
    *, sweeps_root: Path, accept_window: tuple[str, str],
) -> list[Path]:
    """Return the LATEST sweep dir under ``sweeps_root`` whose
    ``regeneration_delta.json`` OR any of its predictor bundles
    is out-of-window. Returns at most one sweep dir.

    The active-round drift scanners
    (``test_regeneration_delta_scanner.py`` +
    ``test_ac5_variance_precondition_scanner.py``) enforce
    commit-pinning ONLY on the LATEST sweep directory (per
    ``ActiveRoundPredictorBundleScanTest``'s scope-to-latest
    semantic: "the latest sweep is the 'current A0 closure
    artifact'; historical sweeps are audit trail"). Older
    sweep dirs are intentionally preserved as historical record
    with their original build_commits frozen — restamping them
    would change their on-disk shas + break the audit trail.

    The wrapper mirrors that scope so it restamps only what the
    scanners enforce. The LATEST sweep dir is selected by
    name-lex-greatest (sweep dirs are timestamp-prefixed). The
    sweep is returned IFF either sub-surface (regen_delta OR
    predictor bundles) is out-of-window — the wrapper's inner
    loop then conditionally refreshes ONLY the drifted
    sub-surface(s). A prior design checked ONLY the regen_delta
    SHA, which silently skipped sweeps where only the bundles
    had drifted (e.g., bundles got restamped via a separate
    code path but the regen_delta did not)."""
    if not sweeps_root.is_dir():
        return []
    candidate_sweeps = [
        d for d in sorted(sweeps_root.iterdir())
        if d.is_dir()
        and (d / "regeneration_delta.json").is_file()
        and (d / "predictor_bundles").is_dir()
    ]
    if not candidate_sweeps:
        return []
    latest_sweep = max(candidate_sweeps, key=lambda d: d.name)
    regen_delta_stale = is_regeneration_delta_out_of_window(
        sweep_dir=latest_sweep, accept_window=accept_window,
    )
    drifted_bundles = discover_drifted_predictor_bundles(
        sweep_dir=latest_sweep, accept_window=accept_window,
    )
    if regen_delta_stale or drifted_bundles:
        return [latest_sweep]
    return []


def is_regeneration_delta_out_of_window(
    *, sweep_dir: Path, accept_window: tuple[str, str],
) -> bool:
    """Return True when ``<sweep_dir>/regeneration_delta.json``
    exists AND its ``regeneration_build_commit`` lies OUTSIDE
    the {HEAD, HEAD^} accept window.

    Computed INDEPENDENTLY from predictor-bundle drift because
    the two surfaces can drift separately. A prior design
    treated the sweep_restamp branch as a single ``no_action``
    when ``discover_drifted_predictor_bundles`` returned empty,
    which silently skipped a stale regeneration_delta when its
    predictor bundles happened to be in-window — leaving the
    sweep partially out-of-window across iterations. The
    current design decouples the two sub-actions; this helper
    drives the regeneration-delta independent check."""
    head, parent = accept_window
    in_window = {head, parent}
    regen_delta = sweep_dir / "regeneration_delta.json"
    if not regen_delta.is_file():
        return False
    try:
        payload = json.loads(regen_delta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    current = payload.get(_REGENERATION_DELTA_FIELD)
    return current not in in_window


def discover_drifted_predictor_bundles(
    *, sweep_dir: Path, accept_window: tuple[str, str],
) -> list[Path]:
    """Return predictor-bundle JSON paths under
    ``<sweep_dir>/predictor_bundles/`` whose
    ``predictor_build_commit`` lies outside the {HEAD, HEAD^}
    window. The refresh helper accepts a comma-separated path list;
    discovering only the drifted ones keeps the helper's output
    focused on actual rewrites."""
    bundles_dir = sweep_dir / "predictor_bundles"
    if not bundles_dir.is_dir():
        return []
    head, parent = accept_window
    in_window = {head, parent}
    out: list[Path] = []
    for p in sorted(bundles_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        current = payload.get(_PREDICTOR_BUNDLE_FIELD)
        if current in in_window:
            continue
        out.append(p)
    return out


def read_prior_source_commits_for_cells(
    *, label_path: Path, cell_keys: list[str],
) -> dict[str, str | None]:
    """Read the label-admitted JSON BEFORE the refresh helper
    runs + return a ``{<arch>/<slug>/<shape_point>:
    <prior_source_commit>}`` map for the requested cell keys.
    Returns ``None`` for any cell key not found or any cell
    that lacks a ``source_commit`` field.

    Used by the wrapper's label-restamp step to capture the
    prior SHA per cell into the per-step status report, so the
    report is self-contained for downstream audit (no `git
    diff` needed to see what SHA each cell held before this
    round's restamp)."""
    if not label_path.is_file():
        return {key: None for key in cell_keys}
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {key: None for key in cell_keys}
    out: dict[str, str | None] = {key: None for key in cell_keys}
    cells = payload.get("cells", []) or []
    for c in cells:
        if not isinstance(c, dict):
            continue
        arch = c.get("arch", "")
        slug = c.get("slug", "")
        shape_point = c.get("shape_point", "")
        cell_key = f"{arch}/{slug}/{shape_point}"
        if cell_key in out:
            out[cell_key] = c.get("source_commit")
    return out


def read_prior_build_commits_for_sweep(
    *, sweep_dir: Path, drifted_bundles: list[Path],
) -> dict[str, Any]:
    """Read the sweep dir's predictor bundles + regeneration_delta
    BEFORE the refresh helper runs + return a structured
    ``{predictor_bundles: {<filename>: <prior_predictor_build_commit>},
    regeneration_delta: <prior_regeneration_build_commit>}`` map.
    Returns ``None`` for any file that's absent / malformed /
    missing the expected field.

    Used by the wrapper's sweep-restamp step to capture prior
    build commits into the per-step status report, so the
    report is self-contained for downstream audit."""
    bundles_out: dict[str, str | None] = {}
    for p in drifted_bundles:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            bundles_out[p.name] = payload.get(
                _PREDICTOR_BUNDLE_FIELD,
            )
        except (OSError, json.JSONDecodeError):
            bundles_out[p.name] = None
    regen_delta_path = sweep_dir / "regeneration_delta.json"
    regen_delta_prior: str | None = None
    try:
        payload = json.loads(
            regen_delta_path.read_text(encoding="utf-8"),
        )
        regen_delta_prior = payload.get(
            _REGENERATION_DELTA_FIELD,
        )
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "predictor_bundles": bundles_out,
        "regeneration_delta": regen_delta_prior,
    }


def restamp_variance_trajectory(
    *, path: Path, head_sha: str,
) -> dict[str, Any]:
    """Rewrite the per-cell ``predictor_build_commit`` entries in
    ``variance_trajectory.json::current_per_cell_predictions`` to
    HEAD when stale. The variance_trajectory file does not have an
    artifact-root build_commit so the standard refresh helper does
    not target it; rewriting per-cell entries in place is the
    audit-annotation-shape-compatible approach.

    Returns a status dict with the count of rewritten entries (0
    means no-op; not an error)."""
    if not path.is_file():
        return {
            "status": "no_action",
            "reason": "variance_trajectory_artifact_absent",
            "path": str(path),
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "reason": (
                f"variance_trajectory_artifact_malformed:"
                f"{type(exc).__name__}"
            ),
            "path": str(path),
        }
    rewrite_counts: dict[str, int] = {}
    entries = data.get("current_per_cell_predictions", []) or []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prior = entry.get(_PREDICTOR_BUNDLE_FIELD)
        if prior == head_sha:
            continue
        rewrite_counts[str(prior)] = (
            rewrite_counts.get(str(prior), 0) + 1
        )
        entry[_PREDICTOR_BUNDLE_FIELD] = head_sha
    total_rewritten = sum(rewrite_counts.values())
    if total_rewritten == 0:
        return {
            "status": "no_action",
            "reason": "all_per_cell_predictor_build_commit_at_head",
            "path": str(path),
            "entries_inspected": len(entries),
        }
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "path": str(path),
        "entries_inspected": len(entries),
        "entries_rewritten": total_rewritten,
        "rewrite_origins": rewrite_counts,
        "rewrote_to": head_sha,
    }


def _invoke_module_main(
    *, module_name: str, argv: list[str] | None = None,
) -> dict[str, Any]:
    """In-process invocation of a tool module's ``main(argv)``. The
    6 v5 emitters + the 2 refresh helpers all expose
    ``def main(argv: list[str] | None = None) -> int``. Returns a
    status dict with exit_code + status (ok | failed).

    `argv` normalization: when the caller passes ``None`` (the
    default), forward an empty list ``[]`` to the child's
    ``main`` rather than ``None``. The child's argparse parser
    treats ``None`` as "use process-level ``sys.argv``", which
    leaks the WRAPPER'S CLI flags (e.g., ``--report-path``)
    into the child parser. Children whose parsers do not
    define ``--report-path`` (every emitter except this
    wrapper) would then ``SystemExit`` on unrecognized
    arguments. Forwarding ``[]`` makes the child parse no
    arguments and use its own defaults, which is the
    intended behavior for default emitter invocations.
    Explicit argv lists from call sites (label / sweep
    refresh helpers) are forwarded verbatim and are not
    affected by this normalization.
    """
    child_argv = [] if argv is None else argv
    try:
        import importlib
        mod = importlib.import_module(module_name)
        exit_code = mod.main(child_argv)
    except SystemExit as exc:
        # argparse / explicit sys.exit
        exit_code = (
            exc.code if isinstance(exc.code, int)
            else (1 if exc.code is not None else 0)
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "module": module_name,
            "exit_code": None,
            "reason": (
                f"{type(exc).__name__}: {exc!s}"
            ),
        }
    return {
        "status": "ok" if exit_code == 0 else "failed",
        "module": module_name,
        "exit_code": exit_code,
    }


# Ordered list of regen step (name, module). Each step's module
# exposes ``main(argv)``; the wrapper invokes them in order, halting
# only if a step exits non-zero (the report still lands). A failed or
# split-unaware fitter step additionally halts the rest of the chain
# fail-closed (downstream emitters must not regenerate committed gate
# artifacts from an unverified head).
_REGEN_STEPS: tuple[tuple[str, str], ...] = (
    (
        "fit_monotonic_calibration_head_v5",
        "tools.local_labels.fit_monotonic_calibration_head_v5",
    ),
    (
        "nugteren2014_rdh_baseline",
        "tools.baseline_runners.nugteren2014_rdh_baseline",
    ),
    (
        "emit_v5_closure_artifacts",
        "tools.local_labels.emit_v5_closure_artifacts",
    ),
    (
        "emit_v5_robustness_artifact",
        "tools.local_labels.emit_v5_robustness_artifact",
    ),
    (
        "run_v5_active_path_comparison",
        "tools.local_labels.run_v5_active_path_comparison",
    ),
    (
        "emit_v5_cohort_staging_manifest",
        "tools.local_labels.emit_v5_cohort_staging_manifest",
    ),
)

# The fitter step's identity + the sealed-cohort invocation it MUST run
# with. These argv values reproduce the canonical standalone sm_86 head
# fit: ledger-pinned admitted bundles, the sealed cohort split
# (calibration+validation fit, holdout excluded), single-arch sm_86
# restriction, and the canonical fitted-head output path.
_FITTER_STEP_NAME = "fit_monotonic_calibration_head_v5"
_COHORT_SPLIT_PATH = (
    "data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json"
)
_REGEN_LEDGER_PATH = "data/local_labels/sm86_cohort_regen_ledger.json"
_SM_86_HEAD_PATH = "data/local_labels/sm_86_v5_head.json"
_COMMITTED_COMPARISON_BUNDLE = (
    "data/local_labels/v5_active_path_comparison_bundle.json"
)

# Per-step argv. Steps not listed run with their module defaults. The
# downstream emitters consume the sm_86 head explicitly (their module
# default is the FROZEN sm_120-era joint head calibration_head_v5.json,
# which this chain must neither read as the sm_86 head nor rewrite).
# The comparison runner gets the committed-bundle opt-in while its
# per-arch head emissions stay on its non-committed scratch defaults.
_STEP_ARGV: dict[str, list[str]] = {
    _FITTER_STEP_NAME: [
        "--admitted-bundles-from-ledger", _REGEN_LEDGER_PATH,
        "--cohort-split", _COHORT_SPLIT_PATH,
        "--arch", "sm_86",
        "--output", _SM_86_HEAD_PATH,
    ],
    "nugteren2014_rdh_baseline": ["--v5-head", _SM_86_HEAD_PATH],
    "emit_v5_closure_artifacts": ["--v5-head", _SM_86_HEAD_PATH],
    "emit_v5_robustness_artifact": ["--v5-head", _SM_86_HEAD_PATH],
    "run_v5_active_path_comparison": [
        "--comparison-out", _COMMITTED_COMPARISON_BUNDLE,
        "--allow-committed-paths",
    ],
}


class FitterSplitConfigError(ValueError):
    """The wrapper's fitter invocation is not split-aware: the argv
    carries no usable ``--cohort-split`` value, so the fit would
    include the sealed holdout cells. Fail-closed — raised BEFORE the
    fitter is invoked."""

    typed_reason = "fitter_step_split_unaware"


def assert_fitter_argv_split_aware(argv: list[str]) -> None:
    """Raise ``FitterSplitConfigError`` unless ``argv`` carries a
    non-empty ``--cohort-split <path>`` value."""
    if "--cohort-split" in argv:
        idx = argv.index("--cohort-split")
        if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
            return
    raise FitterSplitConfigError(
        f"{FitterSplitConfigError.typed_reason}: the fitter argv "
        f"{argv!r} carries no usable --cohort-split value; a "
        "split-unaware invocation would fit on the sealed holdout "
        "cells. Refusing to invoke the fitter."
    )


def _fitter_output_path(argv: list[str]) -> Path:
    """Resolve the fitter's ``--output`` value from its argv (the same
    cwd-relative resolution the fitter itself applies)."""
    if "--output" in argv:
        idx = argv.index("--output")
        if idx + 1 < len(argv):
            return Path(argv[idx + 1])
    return Path(_SM_86_HEAD_PATH)


def verify_emitted_head_split_aware(head_path: Path) -> dict[str, Any]:
    """Post-emission fail-closed check: the head the fitter just wrote
    must record a split-aware calibration+validation fit with the
    holdout-exclusion count. Returns ``{"status": "ok", ...}`` or a
    typed failure dict."""
    try:
        head = json.loads(head_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "reason": f"emitted_head_unreadable:{type(exc).__name__}",
            "head_path": str(head_path),
        }
    fit_scope = head.get("fit_scope") or {}
    if fit_scope.get("split_aware") is not True:
        return {
            "status": "failed",
            "reason": "emitted_head_not_split_aware",
            "head_path": str(head_path),
        }
    if fit_scope.get("fit_splits") != ["calibration", "validation"]:
        return {
            "status": "failed",
            "reason": "emitted_head_fit_splits_mismatch",
            "head_path": str(head_path),
            "fit_splits": fit_scope.get("fit_splits"),
        }
    if not isinstance(fit_scope.get("n_holdout_excluded_from_fit"), int):
        return {
            "status": "failed",
            "reason": "emitted_head_missing_holdout_exclusion_count",
            "head_path": str(head_path),
        }
    return {
        "status": "ok",
        "head_path": str(head_path),
        "split_aware": True,
        "n_fit": fit_scope.get("n_fit"),
        "n_holdout_excluded_from_fit": fit_scope.get(
            "n_holdout_excluded_from_fit",
        ),
        "cohort_split_path": fit_scope.get("cohort_split_path"),
    }


def regenerate_v5_artifacts(
    *,
    report_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Run the full v5 regen + restamp sequence. Always writes the
    report to ``report_path``; returns the report dict for
    testability."""
    head_sha = _git_head_sha()
    parent_sha = _git_parent_sha(head_sha) if head_sha != "unknown" else "unknown"
    accept_window = (head_sha, parent_sha)
    report: dict[str, Any] = {
        "head_at_wrapper_invocation": head_sha,
        "accept_window": list(accept_window),
        "started_at": _now_iso(),
        "dry_run": dry_run,
        "steps": {},
    }
    overall_failed = False

    # Block 1: 6 regen steps in order. The fitter step is gated
    # fail-closed: a split-unaware argv refuses to invoke the fitter,
    # a split-unaware emitted head fails the step, and either failure
    # halts every downstream regen step (the chain must not rewrite
    # committed gate artifacts from an unverified head). The gate runs
    # in dry-run too so a config regression surfaces without executing
    # any emitter.
    halt_chain_reason: str | None = None
    for step_name, module_name in _REGEN_STEPS:
        step_argv = _STEP_ARGV.get(step_name, [])
        if halt_chain_reason is not None:
            report["steps"][step_name] = {
                "status": "skipped_fail_closed",
                "module": module_name,
                "reason": halt_chain_reason,
            }
            continue
        if step_name == _FITTER_STEP_NAME:
            try:
                assert_fitter_argv_split_aware(step_argv)
            except FitterSplitConfigError as exc:
                report["steps"][step_name] = {
                    "status": "failed",
                    "module": module_name,
                    "reason": str(exc),
                }
                overall_failed = True
                halt_chain_reason = (
                    f"fail_closed_after:{step_name}:"
                    f"{FitterSplitConfigError.typed_reason}"
                )
                continue
        if dry_run:
            report["steps"][step_name] = {
                "status": "dry_run_skipped",
                "module": module_name,
                "argv": step_argv,
            }
            continue
        result = _invoke_module_main(
            module_name=module_name, argv=step_argv,
        )
        if step_name == _FITTER_STEP_NAME and result["status"] == "ok":
            head_check = verify_emitted_head_split_aware(
                _fitter_output_path(step_argv),
            )
            result["split_aware_head_check"] = head_check
            if head_check["status"] != "ok":
                result["status"] = "failed"
                result["reason"] = head_check["reason"]
        report["steps"][step_name] = result
        if result["status"] != "ok":
            overall_failed = True
            # Continue with subsequent steps so the report shows all
            # failures, but mark overall failure — EXCEPT after a
            # failed fitter step, which halts the chain fail-closed.
            if step_name == _FITTER_STEP_NAME:
                halt_chain_reason = (
                    f"fail_closed_after:{step_name}:"
                    f"{result.get('reason', 'fitter_step_failed')}"
                )

    # Block 2: label restamp for each drift-managed label file.
    for label_path in _LABEL_FILES:
        step_name = f"label_restamp:{label_path.name}"
        cells = discover_drift_managed_cells(label_path)
        if not cells:
            report["steps"][step_name] = {
                "status": "no_action",
                "reason": "no_drift_managed_cells_in_label_file",
                "label_path": str(label_path),
            }
            continue
        if dry_run:
            report["steps"][step_name] = {
                "status": "dry_run_skipped",
                "label_path": str(label_path),
                "discovered_cells": cells,
            }
            continue
        # Capture prior source_commits BEFORE the refresh helper
        # rewrites them, so the per-step report is self-contained
        # for downstream audit (no `git diff` needed).
        prior_source_commits = read_prior_source_commits_for_cells(
            label_path=label_path, cell_keys=cells,
        )
        argv = [
            "--label-admitted", str(label_path),
            "--cells", ",".join(cells),
        ]
        result = _invoke_module_main(
            module_name=(
                "tools.local_labels.refresh_admitted_label_source_commit"
            ),
            argv=argv,
        )
        result["label_path"] = str(label_path)
        result["discovered_cells"] = cells
        result["prior_source_commits"] = prior_source_commits
        report["steps"][step_name] = result
        if result["status"] != "ok":
            overall_failed = True

    # Block 3: sweep-dir restamp for each drift-managed sweep
    # dir. The two sub-actions (predictor_bundles refresh +
    # regeneration_delta refresh) run INDEPENDENTLY because the
    # two surfaces can drift separately. A sweep was selected
    # iff its regen_delta OR its predictor bundles are
    # out-of-window; the wrapper repairs ONLY the drifted
    # sub-surface(s).
    sweep_dirs = discover_drift_managed_sweep_dirs(
        sweeps_root=_SWEEPS_ROOT, accept_window=accept_window,
    )
    for sweep_dir in sweep_dirs:
        step_name = f"sweep_restamp:{sweep_dir.name}"
        drifted_bundles = discover_drifted_predictor_bundles(
            sweep_dir=sweep_dir, accept_window=accept_window,
        )
        regen_delta_stale = is_regeneration_delta_out_of_window(
            sweep_dir=sweep_dir, accept_window=accept_window,
        )
        if dry_run:
            report["steps"][step_name] = {
                "status": "dry_run_skipped",
                "sweep_dir": str(sweep_dir),
                "drifted_bundle_count": len(drifted_bundles),
                "regeneration_delta_stale": regen_delta_stale,
            }
            continue
        # Capture prior build_commits BEFORE any refresh runs,
        # so the per-step report is self-contained for both
        # sub-surfaces regardless of which gets refreshed.
        prior_build_commits = read_prior_build_commits_for_sweep(
            sweep_dir=sweep_dir, drifted_bundles=drifted_bundles,
        )
        # Sub-action 1: predictor_bundles refresh. Runs ONLY
        # when at least one bundle is drifted; otherwise the
        # sub-result is a structurally-complete no_action so
        # downstream readers see consistent shape.
        if drifted_bundles:
            bundles_argv = [
                "--paths", ",".join(str(p) for p in drifted_bundles),
                "--field", _PREDICTOR_BUNDLE_FIELD,
            ]
            bundles_result = _invoke_module_main(
                module_name=(
                    "tools.local_labels.refresh_sweep_build_commit"
                ),
                argv=bundles_argv,
            )
            bundles_result["sweep_dir"] = str(sweep_dir)
            bundles_result["drifted_bundle_count"] = len(
                drifted_bundles,
            )
            bundles_result["prior_build_commits"] = (
                prior_build_commits["predictor_bundles"]
            )
        else:
            bundles_result = {
                "status": "no_action",
                "reason": "no_drifted_predictor_bundles",
                "sweep_dir": str(sweep_dir),
                "drifted_bundle_count": 0,
                "prior_build_commits": {},
            }
        # Sub-action 2: regeneration_delta refresh. Runs ONLY
        # when its prior `regeneration_build_commit` is
        # out-of-window; otherwise the sub-result is a
        # structurally-complete no_action.
        if regen_delta_stale:
            regen_delta_argv = [
                "--paths",
                str(sweep_dir / "regeneration_delta.json"),
                "--field", _REGENERATION_DELTA_FIELD,
            ]
            regen_delta_result = _invoke_module_main(
                module_name=(
                    "tools.local_labels.refresh_sweep_build_commit"
                ),
                argv=regen_delta_argv,
            )
            regen_delta_result["prior_build_commit"] = (
                prior_build_commits["regeneration_delta"]
            )
        else:
            regen_delta_result = {
                "status": "no_action",
                "reason": (
                    "regeneration_build_commit_in_window"
                ),
                "prior_build_commit": None,
            }
        # Parent status: ok when every sub-action is ok OR
        # no_action; failed when any INVOKED refresh fails.
        sub_statuses = (
            bundles_result["status"],
            regen_delta_result["status"],
        )
        parent_status = (
            "failed" if "failed" in sub_statuses else "ok"
        )
        report["steps"][step_name] = {
            "predictor_bundles": bundles_result,
            "regeneration_delta": regen_delta_result,
            "status": parent_status,
        }
        if parent_status != "ok":
            overall_failed = True

    if not sweep_dirs:
        report["steps"]["sweep_restamp"] = {
            "status": "no_action",
            "reason": "no_drift_managed_sweep_dirs",
        }

    # Block 4: variance_trajectory per-cell entry restamp.
    if dry_run:
        report["steps"]["variance_trajectory_restamp"] = {
            "status": "dry_run_skipped",
            "path": str(_VARIANCE_TRAJECTORY),
        }
    else:
        vt_result = restamp_variance_trajectory(
            path=_VARIANCE_TRAJECTORY, head_sha=head_sha,
        )
        report["steps"]["variance_trajectory_restamp"] = vt_result
        if vt_result["status"] == "failed":
            overall_failed = True

    report["finished_at"] = _now_iso()
    report["overall_status"] = "failed" if overall_failed else "ok"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v5 artifact regeneration + lockstep restamp "
            "from a single invocation."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Discover drift-managed cells + sweep dirs + "
            "variance-trajectory state without executing any "
            "emitter or restamp. The report still lands."
        ),
    )
    parser.add_argument(
        "--report-path", default=str(_DEFAULT_REPORT),
        help=(
            f"Where to write the structured per-step status "
            f"report (default: {_DEFAULT_REPORT})."
        ),
    )
    args = parser.parse_args(argv)
    report = regenerate_v5_artifacts(
        report_path=Path(args.report_path),
        dry_run=args.dry_run,
    )
    return 0 if report["overall_status"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
