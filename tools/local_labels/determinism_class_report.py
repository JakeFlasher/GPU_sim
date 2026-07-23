#!/usr/bin/env python3
"""Per-metric determinism classes + noise floors from a repeatability matrix.

Plan §1C deliverable: classify every collected NCU metric into
  deterministic    — dispersion exactly 0 in EVERY reference-config cell
  quasi_stable     — reference CV <= 0.5% in every cell (max over cells)
  clock_noise      — reference CV <= 2% (timing-class jitter)
  noisy            — reference CV > 2% somewhere
and quantify PROTOCOL sensitivity (relative mean shift base/boost vs the
host-locked reference; cache none vs all) per metric. The per-class floors
are the accuracy-gate denominators for M2/M3: a simulator prediction of a
`deterministic` metric is gated at ~0 tolerance, a `clock_noise` metric at
its measured floor.

Usage:
  .venv/bin/python tools/local_labels/determinism_class_report.py \
      --summary data/annex/repeatability_matrix_full_20260706/matrix_summary.json \
      --out <report>.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repeatability_matrix_runner import cfg_tag  # noqa: E402

# Single source of the config-tag grammar (cfg_tag); restating the literals
# here would fail OPEN (zero matched rows -> all-deterministic) on any
# grammar change.
REF = cfg_tag("none", "all")

# Class boundaries (max reference CV over cells). Stamped into the report so
# frozen reports are self-describing; a metric with NO reference evidence is
# typed `unmeasured_reference`, never classified.
QUASI_STABLE_MAX_CV = 0.005
CLOCK_NOISE_MAX_CV = 0.02


def classify(max_cv: float, any_nonzero: bool) -> str:
    if not any_nonzero:
        return "deterministic"
    if max_cv <= QUASI_STABLE_MAX_CV:
        return "quasi_stable"
    if max_cv <= CLOCK_NOISE_MAX_CV:
        return "clock_noise"
    return "noisy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.summary).read_text())["rows"]
    # (metric, config, slug) -> (mean, relative_cv)
    # dispersion semantics depend on the registry repeatability_rule:
    #   coefficient_of_variation / relative_tolerance -> already relative
    #   stddev_abs -> absolute, normalize by |mean|
    by_key: dict[tuple[str, str, str], tuple[float, float]] = {}
    metrics: set[str] = set()
    for r in rows:
        if r.get("mean") is None or not r.get("metric"):
            continue
        mean = float(r["mean"])
        disp = r.get("dispersion")
        if disp is None:
            continue  # no dispersion evidence -> cannot classify this cell
        disp = float(disp)
        rule = r.get("repeatability_rule")
        if rule in ("coefficient_of_variation", "relative_tolerance"):
            cv = abs(disp)
        elif rule == "stddev_abs":
            cv = abs(disp / mean) if mean != 0.0 else (0.0 if disp == 0.0 else float("inf"))
        else:
            raise SystemExit(f"unknown repeatability_rule {rule!r} for {r['metric']}")
        by_key[(r["metric"], r["config"], r["slug"])] = (mean, cv)
        metrics.add(r["metric"])

    slugs = sorted({k[2] for k in by_key})
    report = {}
    for m in sorted(metrics):
        ref_cvs, ref_cells = [], 0
        any_nonzero = False
        for s in slugs:
            cell = by_key.get((m, REF, s))
            if cell is None:
                continue
            _mean, cv = cell
            ref_cells += 1
            if cv != 0.0:
                any_nonzero = True
            ref_cvs.append(cv)
        # protocol sensitivity: relative mean shift vs reference, max over slugs
        shifts = {}
        for alt_cfg, label in (
            (cfg_tag("base", "all"), "clock_base"),
            (cfg_tag("boost", "all"), "clock_boost"),
            (cfg_tag("none", "none"), "cache_none"),
        ):
            deltas = []
            for s in slugs:
                ref = by_key.get((m, REF, s))
                alt = by_key.get((m, alt_cfg, s))
                if ref and alt and ref[0] != 0.0:
                    deltas.append((alt[0] - ref[0]) / abs(ref[0]))
            if deltas:
                shifts[label] = {
                    "median_rel_shift": round(statistics.median(deltas), 5),
                    "max_abs_rel_shift": round(max(deltas, key=abs), 5),
                }
        if not ref_cvs:
            # FAIL-CLOSED: a metric with zero reference-config evidence must
            # never be classified (least of all as `deterministic`, the
            # strictest gate) — it gets a typed non-class instead.
            report[m] = {
                "determinism_class": "unmeasured_reference",
                "typed_reason": "no_reference_config_cells",
                "reference_cells": 0,
                "reference_max_cv": None,
                "reference_median_cv": None,
                "protocol_sensitivity": shifts,
            }
            continue
        max_cv = max(ref_cvs)
        report[m] = {
            "determinism_class": classify(max_cv, any_nonzero),
            "reference_cells": ref_cells,
            "reference_max_cv": round(max_cv, 6),
            "reference_median_cv": round(statistics.median(ref_cvs), 6),
            "protocol_sensitivity": shifts,
        }

    classes = defaultdict(list)
    if not any(k[1] == REF for k in by_key):
        # FAIL-CLOSED: zero rows matched the reference config — the join is
        # broken (wrong summary file or cfg grammar drift); refuse to emit a
        # report rather than classify everything from missing evidence.
        print(f"determinism_class_report: no rows match reference config "
              f"{REF!r} in {args.summary} — refusing to classify",
              file=sys.stderr)
        return 2

    for m, rec in report.items():
        classes[rec["determinism_class"]].append(m)
    out = {
        "generated_by": "determinism_class_report",
        "reference_config": REF,
        "class_thresholds": {
            "quasi_stable_max_cv": QUASI_STABLE_MAX_CV,
            "clock_noise_max_cv": CLOCK_NOISE_MAX_CV,
        },
        "censoring": (
            "input rows reflect ADMITTED evidence only (registry admission "
            "thresholds CV/rel-tol 0.05, stddev 0.03) — floors above the "
            "admission gate are right-censored; `noisy` is therefore a "
            "(threshold, admission-gate] band, and dispersion statistics "
            "mix stddev-based (CV) and range-based (relative_tolerance) "
            "rules per the registry"
        ),
        "n_metrics": len(report),
        "class_counts": {k: len(v) for k, v in sorted(classes.items())},
        "classes": {k: sorted(v) for k, v in sorted(classes.items())},
        "per_metric": report,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"determinism report: {args.out}")
    for k, v in sorted(classes.items()):
        print(f"  {k}: {len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
