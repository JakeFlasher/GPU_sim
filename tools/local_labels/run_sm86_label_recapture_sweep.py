"""Bulk sm_86 NCU label re-capture sweep driver.

Iterates ``data/local_labels/sm86_cohort_preflight.json`` and
calls ``recapture_cell()`` from
``tools.local_labels.orchestrate_sm86_label_recapture`` on every
cell whose ``binary_present`` flag is true. Emits a committed
per-cell ledger at
``data/local_labels/sm86_label_recapture_sweep_ledger.json`` so
reviewers can audit the sweep without depending on
gitignored per-launch sidecars under
``data/local_labels/sm_86/``.

Per-cell statuses follow a closed enum:

* ``admitted`` — every requested repeat landed a sidecar.
* ``repeats_partial`` — at least one repeat landed a sidecar but
  not all (e.g., NCU returned non-zero on a subset).
* ``recapture_failed`` — zero repeats landed; the orchestrator
  surfaced a typed failure reason.
* ``binary_missing`` — the cell's binary is not on disk; the
  orchestrator was NOT invoked.

The sweep MUST emit the ledger artifact regardless of partial
completion so a mid-sweep interrupt still leaves a useful audit
trail. The ledger is committed; per-launch sidecars and
``.ncu-rep`` files land in the gitignored quarantine path per
CLAUDE.md Rule 1.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

from tools.local_labels.orchestrate_sm86_label_recapture import (  # noqa: E402
    recapture_cell,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PREFLIGHT = (
    _REPO_ROOT / "data" / "local_labels" / "sm86_cohort_preflight.json"
)
_DEFAULT_LEDGER = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "sm86_label_recapture_sweep_ledger.json"
)

_STATUS_ENUM: tuple[str, ...] = (
    "admitted",
    "repeats_partial",
    "recapture_failed",
    "binary_missing",
)


def _classify_status(report: dict[str, Any]) -> str:
    """Map a per-cell ``recapture_cell`` report onto the sweep
    ledger's closed status enum.
    """
    if report["typed_failure_reason"] == "binary_missing_on_disk" or (
        report.get("typed_failure_reason")
        and str(report["typed_failure_reason"]).startswith(
            "binary_missing_on_disk"
        )
    ):
        return "binary_missing"
    if report["repeats_attempted"] == 0:
        return "recapture_failed"
    if report["repeats_succeeded"] == report["repeats_attempted"]:
        return "admitted"
    if report["repeats_succeeded"] > 0:
        return "repeats_partial"
    return "recapture_failed"


def _ledger_entry_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Project a per-cell report onto the ledger's per-cell shape.
    The ledger omits the per-repeat stderr tails so the committed
    artifact stays compact; reviewers who need stderr inspect the
    quarantine path directly.
    """
    return {
        "cell_key": report["cell_key"],
        "status": _classify_status(report),
        "binary_path": report["binary_path"],
        "binary_sha256_observed": report["binary_sha256_observed"],
        "repeats_attempted": report["repeats_attempted"],
        "repeats_succeeded": report["repeats_succeeded"],
        "repeats_failed": report["repeats_failed"],
        "sidecar_paths": report["sidecar_paths"],
        "wall_clock_seconds": report["wall_clock_seconds"],
        "typed_failure_reason": report["typed_failure_reason"],
        "failure_evidence": report.get("failure_evidence"),
        "attempted_at_utc": report.get("attempted_at_utc"),
    }


def run_sweep(
    *,
    preflight_path: Path = _DEFAULT_PREFLIGHT,
    ledger_path: Path = _DEFAULT_LEDGER,
    n_repeats: int = 5,
    warmup: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Iterate the preflight matrix and emit a per-cell sweep
    ledger. Returns the ledger dict.
    """
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    cells = preflight.get("cells", [])
    started_at_utc = (
        _dt.datetime.now(tz=_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    entries: list[dict[str, Any]] = []
    summary_counts: dict[str, int] = {s: 0 for s in _STATUS_ENUM}
    total_wall_clock = 0.0
    for cell in cells:
        slug = cell["slug"]
        shape_point = cell["shape_point"]
        if not cell.get("binary_present"):
            entries.append({
                "cell_key": f"{slug}/{shape_point}",
                "status": "binary_missing",
                "binary_path": cell.get("binary_path"),
                "binary_sha256_observed": None,
                "repeats_attempted": 0,
                "repeats_succeeded": 0,
                "repeats_failed": 0,
                "sidecar_paths": [],
                "wall_clock_seconds": 0.0,
                "typed_failure_reason": (
                    "binary_missing_on_disk:preflight_reports_binary_absent"
                ),
                "attempted_at_utc": started_at_utc,
            })
            summary_counts["binary_missing"] += 1
            continue
        report = recapture_cell(
            slug=slug,
            shape_point=shape_point,
            n_repeats=n_repeats,
            warmup=warmup,
            dry_run=dry_run,
        )
        entry = _ledger_entry_from_report(report)
        entries.append(entry)
        summary_counts[entry["status"]] += 1
        total_wall_clock += float(entry.get("wall_clock_seconds") or 0.0)
    completed_at_utc = (
        _dt.datetime.now(tz=_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    ledger = {
        "schema_version": 1,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "n_repeats": n_repeats,
        "warmup": warmup,
        "dry_run": dry_run,
        "cells_attempted": len(entries),
        "summary": summary_counts,
        "total_wall_clock_seconds": round(total_wall_clock, 3),
        "closed_status_enum": list(_STATUS_ENUM),
        "cells": entries,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-path", type=Path, default=_DEFAULT_PREFLIGHT,
        help=f"Preflight matrix path (default: {_DEFAULT_PREFLIGHT.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--ledger-path", type=Path, default=_DEFAULT_LEDGER,
        help=f"Sweep ledger output path (default: {_DEFAULT_LEDGER.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--n-repeats", type=int, default=5,
        help="NCU repeats per cell (default 5).",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Warmup launches before each repeat (default 1).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Resolve cells + emit ledger WITHOUT invoking NCU. "
            "Useful for verifying the ledger shape before a live "
            "sweep."
        ),
    )
    args = parser.parse_args(argv)
    ledger = run_sweep(
        preflight_path=args.preflight_path,
        ledger_path=args.ledger_path,
        n_repeats=args.n_repeats,
        warmup=args.warmup,
        dry_run=args.dry_run,
    )
    print(
        f"sm_86 label sweep: {ledger['cells_attempted']} cells; "
        f"summary {ledger['summary']}; "
        f"wall_clock {ledger['total_wall_clock_seconds']}s"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
