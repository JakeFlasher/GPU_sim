#!/usr/bin/env python3
"""Counter-repeatability matrix runner (plan §1C — the noise-floor study).

Sweeps each roster cell over the measurement-protocol matrix
    {--clock-control: none | base | boost} x {--cache-control: all | none}
with N repeats per configuration, by invoking ``label_driver.sh`` per
(cell, config, repeat) into a config-partitioned output root, then running
``repeatability_aggregator`` per configuration, and finally emitting one
matrix summary JSON: per (cell, config, metric) -> mean / dispersion.

Those dispersions ARE the per-metric noise floors — the denominators of every
simulator accuracy gate (M2/M3) and a standalone paper section (the missing
GPU analog of the Weaver/McKee counter-determinism studies).

Protocol notes:
- The host is expected clock-locked (nvidia-smi -pm 1; -lgc) — the sidecars
  record the observed state in ``measurement_protocol`` either way, so an
  unlocked run is visible in the evidence, not silent.
- ``clock-control none`` + locked host is the reference configuration; base
  and boost delegate locking to NCU (whose base/boost points differ from the
  host lock), so cross-config deltas measure PROTOCOL sensitivity, not noise.

Usage:
  .venv/bin/python tools/local_labels/repeatability_matrix_runner.py \
      --roster tools/local_labels/repeatability_matrix_roster.json \
      --out-root data/annex/repeatability_matrix_<date> \
      [--repeats 5] [--device sm_86] [--dry-run]

Roster JSON: [{"slug": ..., "binary": ..., "shape_point": ...,
               "app_args": [...], "kernel_name": optional}, ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "tools/local_labels/label_driver.sh"
AGGREGATOR = REPO_ROOT / "tools/local_labels/repeatability_aggregator.py"

CLOCK_MODES = ("none", "base", "boost")
CACHE_MODES = ("all", "none")

# Metric classes spanning T1 counts / T2 cache / T3 timing / stall / SOL —
# one representative per class keeps the matrix cheap while covering every
# determinism class the study reports on.
UNIT_METRICS = (
    "gpc__cycles_elapsed.max",
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
)

REGISTRY = REPO_ROOT / "tools/local_labels/metric_registry.json"

# The driver always collects these six (mandatory hit rates + raw-counter
# fallback pairs); duplicating them in --unit-metric would double-route them
# through the parser's registry-unit path.
DRIVER_ALWAYS_COLLECTED = frozenset({
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "l1tex__t_sectors_lookup_hit.sum",
    "l1tex__t_sectors_lookup_miss.sum",
})


def registry_unit_metrics(arch: str) -> tuple[str, ...]:
    """Full-registry metric set for one arch (plan §1C 'all 61 metrics').

    Eligibility = carrier unit_tagged_metric AND availability != excluded
    AND per_arch_availability[arch] == available, minus the driver's
    always-collected six. requires_runtime_carveout entries are excluded
    here — the matrix runner measures the default (no-carveout) protocol.
    """
    reg = json.loads(REGISTRY.read_text())
    out = []
    for m in reg["metrics"]:
        if m.get("carrier") != "unit_tagged_metric":
            continue
        if m.get("availability") == "excluded":
            continue
        if (m.get("per_arch_availability") or {}).get(arch) != "available":
            continue
        if m["ncu_metric"] in DRIVER_ALWAYS_COLLECTED:
            continue
        out.append(m["ncu_metric"])
    return tuple(out)


def cfg_tag(clock: str, cache: str) -> str:
    return f"cfg=clock_{clock}__cache_{cache}"


def run_cell(cell: dict, *, device: str, out_root: Path, repeats: int,
             clock: str, cache: str, dry_run: bool,
             unit_metrics: tuple[str, ...]) -> int:
    out_dir = out_root / cfg_tag(clock, cache) / "labels"
    failures = 0
    for repeat in range(repeats):
        cmd = [
            str(DRIVER),
            "--device", device,
            "--slug", cell["slug"],
            "--shape-point", cell["shape_point"],
            "--repeat", str(repeat),
            "--launch", "0",
            "--binary", cell["binary"],
            "--warmup", "1",
            "--cache-control", cache,
            "--clock-control", clock,
            "--out-dir", str(out_dir),
        ]
        for m in unit_metrics:
            cmd += ["--unit-metric", m]
        if cell.get("kernel_name"):
            cmd += ["--kernel-name", cell["kernel_name"]]
        cmd += ["--", *cell.get("app_args", [])]
        if dry_run:
            print("DRY:", " ".join(cmd))
            continue
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            failures += 1
            print(
                f"FAIL {cell['slug']} {cfg_tag(clock, cache)} r{repeat}: "
                f"{proc.stderr.strip()[-400:]}",
                file=sys.stderr,
            )
    return failures


def aggregate(out_root: Path, *, device_arch: str, clock: str, cache: str,
              repeats: int) -> dict | None:
    cfg = cfg_tag(clock, cache)
    labels = out_root / cfg / "labels"
    agg_out = out_root / cfg / "agg"
    if not labels.is_dir():
        return None
    proc = subprocess.run(
        [sys.executable, str(AGGREGATOR),
         "--root", str(labels), "--out", str(agg_out),
         "--repeats-required", str(repeats), "--arch", device_arch],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"AGG FAIL {cfg}: {proc.stderr.strip()[-400:]}", file=sys.stderr)
        return None
    admitted = json.loads((agg_out / "label_admitted.json").read_text())
    return admitted


def summarize(admitted_by_cfg: dict) -> dict:
    """Matrix summary: (slug, shape, config) -> per-metric mean/dispersion.

    CENSORING CAVEAT: rows come from ``label_admitted.json`` only, so any
    metric whose dispersion exceeded the registry ADMISSION threshold in a
    cell (CV/rel-tol > 0.05, stddev > 0.03) is absent from that cell's rows.
    The noise floors derived downstream are therefore conditional on
    admission — floors ABOVE the admission gate are right-censored. The
    emitted summary carries this note in ``censoring``.
    """
    rows = []
    for cfg, admitted in admitted_by_cfg.items():
        if not admitted:
            continue
        for cell in admitted.get("cells", []):
            base = {
                "config": cfg,
                "slug": cell.get("slug"),
                "shape_point": cell.get("shape_point"),
                "n_repeats": cell.get("n_repeats") or cell.get("n_sidecars"),
            }
            for k, sd_key in (
                ("cache_hit_rate_l2_sector_pct", "stddev_l2_sector"),
                ("cache_hit_rate_l1tex_sector_pct", "stddev_l1tex_sector"),
            ):
                if cell.get(k) is not None:
                    # The admitted mean is percent-scale (fraction x100) but
                    # the aggregator's stddev_l*_sector stays fraction-scale;
                    # emit both on the SAME percent scale or every downstream
                    # stddev_abs CV is understated by exactly 100x.
                    sd = cell.get(sd_key)
                    rows.append({**base, "metric": k, "mean": cell[k],
                                 "dispersion": sd * 100.0 if sd is not None else None,
                                 "repeatability_rule": "stddev_abs"})
            for m in cell.get("unit_tagged_metrics", []) or []:
                rows.append({
                    **base,
                    "metric": m.get("ncu_metric"),
                    "mean": m.get("value"),
                    "dispersion": m.get("dispersion"),
                    "repeatability_rule": m.get("repeatability_rule"),
                    "unit": m.get("unit"),
                })
    return {
        "generated_by": "repeatability_matrix_runner",
        "censoring": (
            "rows reflect ADMITTED evidence only; per-metric dispersions "
            "above the registry admission thresholds (CV/rel-tol 0.05, "
            "stddev 0.03) are right-censored out of their cells"
        ),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roster", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="sm_86")
    ap.add_argument("--clock-modes", default=",".join(CLOCK_MODES))
    ap.add_argument("--cache-modes", default=",".join(CACHE_MODES))
    ap.add_argument(
        "--metrics-from-registry", action="store_true",
        help="collect the FULL registry metric set for --device (plan §1C "
             "full-tier sweep) instead of the 6-representative default",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    unit_metrics = (
        registry_unit_metrics(args.device) if args.metrics_from_registry
        else UNIT_METRICS
    )

    roster = json.loads(Path(args.roster).read_text())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    clocks = [c for c in args.clock_modes.split(",") if c]
    caches = [c for c in args.cache_modes.split(",") if c]

    total_fail = 0
    for clock in clocks:
        for cache in caches:
            for cell in roster:
                total_fail += run_cell(
                    cell, device=args.device, out_root=out_root,
                    repeats=args.repeats, clock=clock, cache=cache,
                    dry_run=args.dry_run, unit_metrics=unit_metrics,
                )
    if args.dry_run:
        return 0

    admitted_by_cfg = {}
    agg_failures = []
    for clock in clocks:
        for cache in caches:
            cfg = cfg_tag(clock, cache)
            admitted_by_cfg[cfg] = aggregate(
                out_root, device_arch=args.device, clock=clock, cache=cache,
                repeats=args.repeats,
            )
            if admitted_by_cfg[cfg] is None:
                # FAIL-CLOSED: a config whose aggregation failed must be
                # distinguishable from one never run — typed record + exit 1.
                agg_failures.append(cfg)

    summary = summarize(admitted_by_cfg)
    summary["capture_failures"] = total_fail
    summary["aggregation_failures"] = agg_failures
    summary_path = out_root / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"matrix summary: {summary_path} ({len(summary['rows'])} rows, "
          f"{total_fail} capture failures, "
          f"{len(agg_failures)} aggregation failures)")
    return 1 if (total_fail or agg_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
