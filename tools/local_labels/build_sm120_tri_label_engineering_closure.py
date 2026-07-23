"""Assemble the sm_120 tri-label DURATION-leg engineering-closure artifact.

The frozen tri-label roster requires every cell to carry trace + L1 + L2 +
duration. The Goldilocks feasibility sweep
(`data/local_labels/duration_trace_goldilocks_sweep.json`) shows the NVBit trace
(byte-bounded -> needs SMALL kernels) and the CUPTI dual-source duration
(reconciles -> needs LARGE kernels) have DISJOINT size windows, so no single
cell can carry both. This generator derives the terminal engineering-closure
artifact from the committed evidence (the sweep, the production duration cohort,
`duration_admitted.json`, and the live tri-label coverage), keeping MEASURED and
INFERRED facts separate. The verdict for the per-cell four-source DURATION leg is
`engineering_closure_refuted`; the immutable AC is recorded as unmeetable for
that leg, NOT weakened.

Run (dependency-light; no jsonschema):
    uv run python tools/local_labels/build_sm120_tri_label_engineering_closure.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import sm120_tri_label_roster_validator as rv  # noqa: E402

_DATA = _REPO_ROOT / "data" / "local_labels"
SWEEP_PATH = _DATA / "duration_trace_goldilocks_sweep.json"
COHORT_PATH = _DATA / "sm120_duration_cohort.json"
ADMITTED_PATH = _DATA / "duration_admitted.json"
ROSTER_PATH = _DATA / "sm120_tri_label_roster.json"
OUT_PATH = _DATA / "sm120_tri_label_engineering_closure.json"

CLOSURE_ID = "sm120_tri_label_duration_leg_closure_v1"
CLOSED_LEG = "duration"
VERDICT = "engineering_closure_refuted"
# The production reconcile floor lists the FLOOR_COUNT smallest admitted cells
# (fixed cardinality so the validator can compare to admitted_sorted[:FLOOR_COUNT]
# exactly and reject truncation).
FLOOR_COUNT = 3


def _resolve_source_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                              capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "0" * 40
    sha = proc.stdout.strip()
    return sha if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha) else "0" * 40


def _tri_label_coverage_counts() -> dict:
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    presence = rv.presence_from_artifacts(roster, _REPO_ROOT)
    counts = {src: 0 for src in rv.TRI_LABEL_SOURCES}
    for sources in presence.values():
        for s in sources:
            if s in counts:
                counts[s] += 1
    gaps = rv.tri_label_coverage(roster, presence)
    counts["four_source_gaps"] = len(gaps)
    counts["total_cells"] = len(roster.get("cells", []))
    return counts


def build() -> dict:
    sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    cohort = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    admitted = json.loads(ADMITTED_PATH.read_text(encoding="utf-8"))

    reconciling_slugs = sorted({str(c["slug"]) for c in cohort.get("cells", [])})

    # MEASURED facts from the sweep, kept in separate first-class sections:
    # within-budget (real bytes <= budget), over-budget (real bytes > budget),
    # and timed-out (UNMEASURED bytes). A timeout is never folded into a measured
    # section.
    measured_within_budget = []
    measured_over_budget = []
    timed_out = []
    skipped = []
    swept_slugs = set()

    def _point(p):
        return {"slug": p["slug"], "size": p["size"], "trace_bytes": p["trace"].get("trace_bytes"),
                "duration_ms": p.get("duration", {}).get("duration_ms"),
                "duration_reconciled": p.get("duration", {}).get("reconciled")}

    for p in sweep.get("points", []):
        if "size" not in p:
            continue
        swept_slugs.add(p["slug"])
        tr = p.get("trace", {})
        status = tr.get("trace_status")
        if status == "measured" and tr.get("byte_budget_status") == "within_budget":
            measured_within_budget.append(_point(p))
        elif status == "measured" and tr.get("byte_budget_status") == "over_budget":
            measured_over_budget.append(_point(p))
        elif status == "timeout":
            timed_out.append({"slug": p["slug"], "size": p["size"], "trace_status": "timeout"})
        elif status in ("skipped_after_timeout", "skipped_after_measured_over_budget"):
            skipped.append({"slug": p["slug"], "size": p["size"], "trace_status": status})
    # The measured trace-admissible ceiling = largest within-budget trace.
    ceiling = max(measured_within_budget, key=lambda m: m["trace_bytes"], default=None)

    # Production reconcile floor from the committed admitted cohort.
    admitted_cells = sorted(
        ({"slug": c["slug"], "shape_point": c["shape_point"],
          "median_duration_ns": c.get("median_duration_ns")} for c in admitted.get("cells", [])),
        key=lambda c: (c["median_duration_ns"] or 0),
    )
    smallest_reconciling = admitted_cells[:FLOOR_COUNT]

    coverage = _tri_label_coverage_counts()

    return {
        "schema_version": 1,
        "closure_id": CLOSURE_ID,
        "arch": "sm_120",
        "source_commit": _resolve_source_commit(),
        "closed_leg": CLOSED_LEG,
        "verdict": VERDICT,
        "trace_budget_mib": sweep.get("trace_budget_mib", 64),
        "verified_live_sm120_proof": sweep.get("verified_live_sm120_proof", {}),
        "evidence_artifact_paths": {
            "goldilocks_sweep": "data/local_labels/duration_trace_goldilocks_sweep.json",
            "duration_cohort": "data/local_labels/sm120_duration_cohort.json",
            "duration_admitted": "data/local_labels/duration_admitted.json",
            "tri_label_roster": "data/local_labels/sm120_tri_label_roster.json",
        },
        "reconciling_slugs": reconciling_slugs,
        "reconciling_slugs_swept": sorted(swept_slugs),
        "measured": {
            "trace_within_budget_points": sorted(
                measured_within_budget, key=lambda m: (m["slug"], m["size"])),
            "trace_over_budget_points": sorted(
                measured_over_budget, key=lambda m: (m["slug"], m["size"])),
            "trace_admissible_ceiling": ceiling,
            "note": (
                "trace_within_budget_points and trace_over_budget_points are BOTH real "
                "byte measurements (trace_status=measured in the sweep). The GEMM n=768 "
                "traces are MEASURED over budget (~120-209 MiB > 64 MiB), not timeouts. "
                "The ceiling is the largest within-budget trace; every within-budget "
                "size is sub-millisecond. Timed-out traces are tracked separately under "
                "timed_out_traces with UNMEASURED byte size."
            ),
        },
        "timed_out_traces": sorted(timed_out, key=lambda t: (t["slug"], t["size"])),
        "timed_out_note": (
            "These traces (the attention slugs at n=512) did not complete within the "
            "per-trace wall cap; their byte sizes are UNMEASURED and are NOT claimed as "
            "measured byte over-budget (distinct from the MEASURED over-budget GEMM "
            "n=768 rows above)."
        ),
        "skipped_traces": sorted(skipped, key=lambda s: (s["slug"], s["size"])),
        "production_duration_reconcile_floor": {
            "evidence": ["data/local_labels/sm120_duration_cohort.json",
                         "data/local_labels/duration_admitted.json"],
            "smallest_reconciling_cells": smallest_reconciling,
            "note": (
                "Authoritative dual-source reconcile floor from the committed "
                "production cohort (--shape-point + override path): the smallest "
                "reconciling cells run several ms (n~2048); a kernel must run "
                ">~2-3ms for host enqueue/sync overhead to fall under 5%."
            ),
        },
        "current_tri_label_coverage": coverage,
        "windows_disjoint": True,
        "rationale": (
            "The NVBit trace is byte-budget-admissible only at small (sub-millisecond) "
            "kernels: GEMM variants are within budget at n<=512 (~5-60 MiB) and MEASURED "
            "over budget at n=768 (~120-209 MiB > 64 MiB); the attention slugs do not "
            "complete a trace at n=512 within the wall cap (timeout, byte size unmeasured). "
            "Either way no trace-budget-admissible size exceeds ~512^3, while the "
            "CUPTI dual-source duration reconciles only at large (>~2-3ms) kernels. The "
            "two size windows are disjoint, so a single (arch, slug, shape_point) cell "
            "cannot carry BOTH a trace and a reconciling duration. The per-cell "
            "four-source (trace+L1+L2+duration) requirement is therefore unmeetable for "
            "the DURATION leg. This is a structural property of the two collection "
            "methods, not of the slug set. The >=50-cell cohort proceeds as a BI-label "
            "(trace+L1+L2) base; the duration label lives in the SEPARATE duration cohort "
            "(sm120_duration_cohort.json) and feeds the duration predictor. The tri-label "
            "roster validator stays fail-closed and still reports four-source gaps; this "
            "artifact does NOT claim those gaps are satisfied."
        ),
        "claims_four_source_roster_satisfied": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    for p in (SWEEP_PATH, COHORT_PATH, ADMITTED_PATH, ROSTER_PATH):
        if not p.is_file():
            print(f"missing required evidence artifact: {p}")
            return 2
    doc = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: verdict={doc['verdict']} closed_leg={doc['closed_leg']} "
          f"slugs={len(doc['reconciling_slugs'])} gaps={doc['current_tri_label_coverage']['four_source_gaps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
