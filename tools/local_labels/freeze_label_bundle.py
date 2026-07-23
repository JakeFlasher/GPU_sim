#!/usr/bin/env python3
"""Freeze the M1 sm_86 label+trace bundle (plan gate M1).

Emits ONE tracked manifest that pins, per roster cell:
- the 5 v2.1-protocol label sidecars (sha256 each) + the aggregator's
  admitted means for the headline gate metrics;
- the NVBit trace identity (dynamic_trace.pb sha256 + stats.csv sha256 +
  admitted record count + binary_sha256 where recorded);
- the protocol stamp (v2.1: host -lgc 1350 + -lmc 10501 + --clock-control
  none + --cache-control all) and the registry/git provenance.

The manifest is the drift detector: any re-capture, re-build, or protocol
change shows up as a hash mismatch against this file. Evidence itself stays
in the machine-local annex; only identities are tracked.

Usage:
  .venv/bin/python tools/local_labels/freeze_label_bundle.py \
      --labels-root data/annex/repeatability_matrix_v21_ref_20260706 \
      --trace-manifest data/annex/m1_trace_capture_20260706/capture_manifest.json \
      --out eval/frozen_labels/sm86_m1_bundle_v21.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_CFG = "cfg=clock_none__cache_all"
HEADLINE = (
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--trace-manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels_root = REPO_ROOT / args.labels_root
    admitted = json.loads(
        (labels_root / REF_CFG / "agg" / "label_admitted.json").read_text()
    )
    excluded = json.loads(
        (labels_root / REF_CFG / "agg" / "label_excluded.json").read_text()
    )
    traces = json.loads((REPO_ROOT / args.trace_manifest).read_text())
    trace_by_slug = {c["slug"]: c for c in traces["cells"]}

    head_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=REPO_ROOT,
    )
    if head_proc.returncode != 0 or not head_proc.stdout.strip():
        print("freeze_label_bundle: git rev-parse HEAD failed — refusing to "
              "freeze without commit provenance", file=sys.stderr)
        return 2
    head = head_proc.stdout.strip()

    cells = []
    problems = []
    for cell in admitted.get("cells", []):
        slug = cell["slug"]
        label_dir = (labels_root / REF_CFG / "labels" / cell["arch"] /
                     slug / cell["shape_point"])
        sidecars = sorted(label_dir.glob("label_*_*.json"))
        sidecars = [p for p in sidecars if "run_manifest" not in p.name]
        if len(sidecars) != 5:
            problems.append(f"{slug}: {len(sidecars)} label sidecars != 5")

        # PROTOCOL VERIFICATION (never stamp what the evidence doesn't say):
        # every sidecar's measurement_protocol must match the pinned v2.1
        # reference; the bundle refuses to certify labels captured under any
        # other clock state.
        headline: dict = {}
        for p in sidecars:
            sc = json.loads(p.read_text())
            mp = sc.get("measurement_protocol") or {}
            if mp.get("ncu_clock_control") != "none":
                problems.append(
                    f"{slug}/{p.name}: ncu_clock_control="
                    f"{mp.get('ncu_clock_control')!r} != 'none'")
            if mp.get("host_persistence_mode") != "Enabled":
                problems.append(
                    f"{slug}/{p.name}: host persistence "
                    f"{mp.get('host_persistence_mode')!r} != Enabled")
            if mp.get("host_sm_clock_mhz") != 1350:
                problems.append(
                    f"{slug}/{p.name}: host_sm_clock_mhz="
                    f"{mp.get('host_sm_clock_mhz')!r} != 1350")
        # Headline means come straight from the admitted cell's own
        # unit_tagged_metrics (same evidence whose sidecar hashes are pinned
        # below) — a separate matrix_summary.json could be stale or from a
        # different roster.
        for m in cell.get("unit_tagged_metrics", []) or []:
            if m.get("ncu_metric") in HEADLINE:
                headline[m["ncu_metric"]] = {
                    "mean": m.get("value"),
                    "dispersion": m.get("dispersion"),
                    "repeatability_rule": m.get("repeatability_rule"),
                }
        missing_headline = set(HEADLINE) - set(headline)
        if missing_headline:
            problems.append(
                f"{slug}: headline metrics missing from admitted cell: "
                f"{sorted(missing_headline)}")

        tr = trace_by_slug.get(slug)
        if tr is None or not tr.get("ok") or not tr.get("contract_ok"):
            problems.append(f"{slug}: trace missing or contract-invalid")
            tr = tr or {}
        trace_folder = Path(tr["traces_folder"]) if tr.get("traces_folder") else None
        trace_ident = {}
        if trace_folder and trace_folder.is_dir():
            pb = trace_folder / "dynamic_trace.pb"
            st = trace_folder / "stats.csv"
            # Payload identity: the top-level pb is a tiny header (the bulk
            # lives in threadblocks/*.pb) — pin a sorted (relpath, size)
            # manifest hash of the payload tree, O(stat) not O(read).
            tb_root = trace_folder / "threadblocks"
            tb_entries = sorted(
                (str(f.relative_to(tb_root)), f.stat().st_size)
                for f in tb_root.rglob("*.pb")
            ) if tb_root.is_dir() else []
            tb_manifest_sha = hashlib.sha256(
                json.dumps(tb_entries).encode()
            ).hexdigest()
            trace_ident = {
                "traces_folder": str(trace_folder.relative_to(REPO_ROOT))
                if trace_folder.is_relative_to(REPO_ROOT) else str(trace_folder),
                "dynamic_trace_pb_sha256": _sha256(pb) if pb.is_file() else None,
                "stats_csv_sha256": _sha256(st) if st.is_file() else None,
                "threadblocks_manifest_sha256": tb_manifest_sha,
                "threadblocks_file_count": len(tb_entries),
                "records_admitted": tr.get("records_admitted"),
                "binary_sha256": tr.get("binary_sha256"),
                "via": tr.get("via"),
            }
            # Label<->trace binary identity must hold, and absence is a
            # problem, not a null to pass through silently.
            if not tr.get("binary_sha256"):
                problems.append(f"{slug}: trace records no binary_sha256")
            elif cell.get("binary_sha256") and \
                    tr["binary_sha256"] not in cell["binary_sha256"]:
                problems.append(
                    f"{slug}: trace binary_sha256 != label binary_sha256")
        cells.append({
            "slug": slug,
            "shape_point": cell["shape_point"],
            "arch": cell["arch"],
            "binary_sha256": cell.get("binary_sha256"),
            "n_repeats": cell.get("n_repeats"),
            "label_sidecar_sha256s": {p.name: _sha256(p) for p in sidecars},
            "headline_admitted": headline,
            "cache_hit_rate_l2_sector_pct": cell.get("cache_hit_rate_l2_sector_pct"),
            "cache_hit_rate_l1tex_sector_pct": cell.get("cache_hit_rate_l1tex_sector_pct"),
            "trace": trace_ident,
        })

    bundle = {
        "bundle_id": "sm86_m1_bundle_v21",
        "gate": "M1",
        "arch": "sm_86",
        "sku": "rtx_3090_ti",
        "protocol": {
            "version": "v2.1",
            "host": "nvidia-smi -pm 1; -lgc 1350; -lmc 10501",
            "ncu": "--clock-control none --cache-control all --replay-mode kernel",
            "driver_version": "610.43.02",
        },
        "source_commit": head,
        "labels_root": args.labels_root,
        "trace_manifest": args.trace_manifest,
        "n_cells": len(cells),
        "excluded_cells": [
            {
                "slug": c.get("slug"),
                "typed_reason": c.get("typed_reason"),
                "detail": c.get("detail"),
            }
            for c in excluded.get("cells", [])
        ],
        "problems": problems,
        "cells": cells,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=1, sort_keys=True) + "\n")
    print(f"frozen bundle: {out} — {len(cells)} cells, "
          f"{len(problems)} problems")
    for p in problems:
        print(f"  PROBLEM: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
