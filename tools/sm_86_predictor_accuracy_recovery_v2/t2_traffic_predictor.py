#!/usr/bin/env python3
"""T2 miss-stream traffic predictor: composed from COMMITTED
artifacts only (offline; zero label-fitted parameters).

Three traffic primitives per admitted cell, each scored against the
live-captured T2 stream labels by closed-identity join:

- ``read_sectors`` (label
  ``l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum``):
  predicted = the EXACT T1 replay count of global-load sectors MINUS
  the rebuilt warp-pool walker's hit count — the sector stream that
  MISSES (or bypasses) L1 and crosses the xbar. Gates: the T1
  ld-sector emission is scored; the walker row is scalar; the cell
  has NO local-class traffic (local lds/sts share the walker's hit
  pool and would contaminate the subtraction); LDGSTS cells fail
  closed upstream (their T1 surface is typed async-copy).
- ``write_sectors`` (label
  ``l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum``):
  predicted = the EXACT T1 global-store sector count — the
  WRITE-POLICY FILTER: global stores are write-through with no
  write-allocate on this device family, so every store sector
  crosses to L2. Gates: the T1 st-sector emission is scored AND the
  cell has no local-class traffic — the label's ``mem_lg`` scope is
  LOCAL+GLOBAL, and local store sectors (spills) cross the xbar too,
  so a local-mixed cell's global-only composition is un-attributable
  (diagnostic), never silently under-predicted.
- ``l2_requests`` (label ``lts__t_requests_srcunit_tex.sum``, the
  L2 requests sourced from the L1TEX unit): predicted =
  (predicted read sectors + predicted write sectors) /
  SECTORS_PER_L2_REQUEST (the 128B request granularity under full
  coalescing) PLUS the EXACT T1 atomic and reduction request counts
  (atomics reach L2 at request granularity outside the sector
  streams). Gates: BOTH upstream primitives scored AND the atom/red
  T1 counts scored. Partial coalescing and atomic-variant L2
  retries land as honest directional error.

Fail-closed scoring: a traffic count whose label was never captured
— every DRAM-level row in this corpus (typed
``label_count_unstable`` with raw repeat evidence) plus the standing
capture-unstable cells — is ``label_unavailable`` and contributes to
NO statistic (no vacuous pass). A primitive whose gate is unmet is
``diagnostic`` and BARS promotion of any hit-rate derived from it
(the pp-bucket report marks those cells barred).

Directional targets (DEC-7): p50 <= 10% / p90 <= 25% relative count
error with a declared denominator floor; the hit-rate pp-bucket
report (p50 <= 5pp / p90 <= 12pp, low/mid/high label buckets) is
computed from the COMMITTED warp-pool re-score pairs — no label is
re-read and no holdout governance budget is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

from warp_pool_l1_rescore import (  # noqa: E402
    RescoreGovernanceError,
    assert_zero_fitted_parameters,
)

DEFAULT_LABELS = (
    _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
)
DEFAULT_T2_EXCLUSIONS = (
    _REPO_ROOT / "data" / "local_labels"
    / "t2_stream_label_exclusions_sm_86.json"
)
DEFAULT_EMISSIONS = (
    _REPO_ROOT / "data" / "local_labels"
    / "replay_count_emissions_sm_86.json"
)
DEFAULT_WALKER = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "warp_pool_l1_per_admitted_cell.json"
)
DEFAULT_RESCORE = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "warp_pool_l1_rescore_sm_86.json"
)
DEFAULT_OUT = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "t2_traffic_prediction_sm_86.json"
)
DEFAULT_SCHEMA = _HERE / "t2_traffic_prediction.schema.json"

_LBL_READ = "l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld.sum"
_LBL_WRITE = "l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st.sum"
_LBL_L2REQ = "lts__t_requests_srcunit_tex.sum"
_T1_LD = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
_T1_ST = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
_T1_LOCAL = (
    "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
)
_T1_ATOM = "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"
_T1_RED = "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"

# FROZEN structural configuration — nothing here was fit to any
# label; its sha256 binds the artifact.
PREDICTOR_CONFIG: dict = {
    "model": "t2_traffic_composition_v1",
    "read_sectors": "t1_ld_sectors_minus_walker_hits",
    "write_sectors": "t1_st_sectors_write_through_no_allocate",
    "l2_requests": (
        "(read+write)_sectors_over_sectors_per_request"
        "_plus_t1_atom_red_requests"
    ),
    # 128B L2 request granularity on this device family (4 x 32B
    # sectors) — an architectural constant, not a tuned value.
    "sectors_per_l2_request": 4,
    "sectors_per_l2_request_provenance": (
        "128B line / 32B sector architectural granularity (the same "
        "SECTORS_PER_LINE=4 the L1 walkers pin); corroborated by "
        "exact-fit composition on full-coalescing cells (e.g. "
        "global_scalar_32b_copy (4096+4096)/4 == label 2048; "
        "carveout cell rel ~8e-06)"
    ),
    "rel_denominator_floor_sectors": 64.0,
    "directional_targets": {"p50": 0.10, "p90": 0.25},
    "hit_rate_pp_targets": {"p50_pp": 5.0, "p90_pp": 12.0},
    "hit_rate_buckets": {
        "low": [0.0, 0.3333333333333333],
        "mid": [0.3333333333333333, 0.6666666666666666],
        "high": [0.6666666666666666, 1.0001],
    },
    "non_vacuity_floor": {
        "min_scored_pairs": 10,
        "min_distinct_slugs": 5,
    },
    "label_fitted_parameters": [],
}


def config_digest() -> str:
    return hashlib.sha256(
        json.dumps(PREDICTOR_CONFIG, sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _identity_key(row: dict) -> tuple:
    return (
        row["slug"], row["shape_point"],
        row["binary_sha256"], row["kernel_signature_sha256"],
    )


def _t1_value(em_cell: dict | None, metric: str) -> float | None:
    """The scored T1 replay-count value, or None when the cell or
    metric is typed (fail-closed: a typed T1 surface gates the
    primitive diagnostic, never a guessed value)."""
    if em_cell is None:
        return None
    if em_cell.get("cell_status", {}).get("status") != "emitted":
        return None
    m = (em_cell.get("metrics") or {}).get(metric)
    if not m or m.get("status") != "scored":
        return None
    return float(m["value"])


def _local_traffic_present(em_cell: dict | None) -> bool | None:
    """True/False when the local-class T1 counts decide the mixture
    gate; None when they are typed (undecidable -> diagnostic)."""
    if em_cell is None:
        return None
    if em_cell.get("cell_status", {}).get("status") != "emitted":
        return None
    total = 0.0
    for metric in _T1_LOCAL:
        m = (em_cell.get("metrics") or {}).get(metric)
        if not m or m.get("status") != "scored":
            return None
        total += float(m["value"])
    return total > 0


def _primitive_row(
    *,
    name: str,
    label_metric: str,
    label_value: float | None,
    label_status: str,
    predicted: float | None,
    gates: dict[str, bool | None],
) -> dict:
    """One (cell, primitive) row under the closed status
    vocabulary. Label first (fail-closed: no label, no score), then
    gates, then the scored pair."""
    floor = float(
        PREDICTOR_CONFIG["rel_denominator_floor_sectors"])
    if label_status != "stable":
        return {
            "primitive": name,
            "label_metric": label_metric,
            "status": "label_unavailable",
            "status_reason": label_status,
            "predicted": predicted,
            "label": None,
            "rel_error": None,
            "gates": gates,
        }
    unmet = sorted(
        g for g, ok in gates.items() if ok is not True)
    if unmet or predicted is None:
        return {
            "primitive": name,
            "label_metric": label_metric,
            "status": "diagnostic",
            "status_reason": "gate_unmet:" + ",".join(unmet),
            "predicted": predicted,
            "label": label_value,
            "rel_error": None,
            "gates": gates,
        }
    rel = abs(predicted - label_value) / max(label_value, floor)
    return {
        "primitive": name,
        "label_metric": label_metric,
        "status": "scored",
        "status_reason": None,
        "predicted": predicted,
        "label": label_value,
        "rel_error": rel,
        "gates": gates,
    }


def _label_status_map(t2_exclusions: dict) -> dict[tuple, str]:
    """(slug, shape_point, ncu_metric) -> typed reason for every
    per-metric T2 exclusion; cell-level terminal rows map all three
    label metrics."""
    out: dict[tuple, str] = {}
    for ex in t2_exclusions.get("exclusions", []):
        reason = ex.get("typed_reason", "label_unavailable")
        if ex.get("ncu_metric"):
            out[(ex["slug"], ex["shape_point"], ex["ncu_metric"])] \
                = reason
        else:
            for metric in (_LBL_READ, _LBL_WRITE, _LBL_L2REQ):
                out[(ex["slug"], ex["shape_point"], metric)] = reason
    return out


def _pctile(sorted_values: list[float], q: float) -> float:
    idx = max(0, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[idx]


def _stats(rels: list[float]) -> dict:
    if not rels:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    s = sorted(rels)
    return {
        "n": len(s),
        "p50": _pctile(s, 0.50),
        "p90": _pctile(s, 0.90),
        "max": s[-1],
    }


def build_prediction(
    *,
    labels_path: Path,
    t2_exclusions_path: Path,
    emissions_path: Path,
    walker_path: Path,
    rescore_path: Path,
) -> dict:
    assert_zero_fitted_parameters(PREDICTOR_CONFIG)

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    t2_excl = json.loads(
        t2_exclusions_path.read_text(encoding="utf-8"))
    emissions = json.loads(
        emissions_path.read_text(encoding="utf-8"))
    walker = json.loads(walker_path.read_text(encoding="utf-8"))
    rescore = json.loads(rescore_path.read_text(encoding="utf-8"))

    em_by = {}
    for c in emissions["cells"]:
        ident = c["identity"]
        em_by.setdefault((
            ident["slug"], ident["shape_point"],
            ident["binary_sha256"],
            ident["kernel_signature_sha256"]), c)
    wp_by = {}
    for c in walker["cells"]:
        wp_by.setdefault(_identity_key(c), c)
    unstable = _label_status_map(t2_excl)

    cells_out: list[dict] = []
    for cell in labels["cells"]:
        if cell.get("arch") != "sm_86":
            continue
        key = _identity_key(cell)
        slug, shape_point = key[0], key[1]
        utm = {
            e["ncu_metric"]: float(e["value"])
            for e in cell.get("unit_tagged_metrics") or []
        }
        em_cell = em_by.get(key)
        wp_cell = wp_by.get(key)

        def label_of(metric: str) -> tuple[float | None, str]:
            if metric in utm:
                return utm[metric], "stable"
            reason = unstable.get((slug, shape_point, metric))
            return None, reason or "label_not_captured"

        t1_ld = _t1_value(em_cell, _T1_LD)
        t1_st = _t1_value(em_cell, _T1_ST)
        local_mix = _local_traffic_present(em_cell)
        walker_scalar = (
            wp_cell is not None
            and wp_cell.get("result_kind") == "scalar"
        )
        hits = (
            float(wp_cell["evidence"]["counters"]["hit_count"])
            if walker_scalar else None
        )

        # read_sectors
        read_label, read_status = label_of(_LBL_READ)
        read_gates = {
            "t1_ld_sectors_scored": t1_ld is not None,
            "walker_scalar": walker_scalar,
            "no_local_class_traffic": (
                True if local_mix is False
                else (False if local_mix is True else None)
            ),
        }
        read_pred = (
            t1_ld - hits
            if (t1_ld is not None and hits is not None) else None
        )
        read_row = _primitive_row(
            name="read_sectors", label_metric=_LBL_READ,
            label_value=read_label, label_status=read_status,
            predicted=read_pred, gates=read_gates,
        )

        # write_sectors (the label's mem_lg scope covers LOCAL too,
        # so the global-only composition needs the same
        # local-mixture gate as the read primitive)
        write_label, write_status = label_of(_LBL_WRITE)
        write_gates = {
            "t1_st_sectors_scored": t1_st is not None,
            "no_local_class_traffic": (
                True if local_mix is False
                else (False if local_mix is True else None)
            ),
        }
        write_row = _primitive_row(
            name="write_sectors", label_metric=_LBL_WRITE,
            label_value=write_label, label_status=write_status,
            predicted=t1_st, gates=write_gates,
        )

        # l2_requests (gated on both upstream primitives + the
        # exact atomic/reduction request counts)
        l2_label, l2_status = label_of(_LBL_L2REQ)
        t1_atom = _t1_value(em_cell, _T1_ATOM)
        t1_red = _t1_value(em_cell, _T1_RED)
        upstream_ok = (
            read_row["status"] == "scored"
            and write_row["status"] == "scored"
            and t1_atom is not None
            and t1_red is not None
        )
        l2_gates = {
            "read_primitive_scored":
                read_row["status"] == "scored",
            "write_primitive_scored":
                write_row["status"] == "scored",
            "t1_atom_red_scored": (
                t1_atom is not None and t1_red is not None
            ),
        }
        l2_pred = (
            (read_row["predicted"] + write_row["predicted"])
            / float(PREDICTOR_CONFIG["sectors_per_l2_request"])
            + t1_atom + t1_red
            if upstream_ok else None
        )
        l2_row = _primitive_row(
            name="l2_requests", label_metric=_LBL_L2REQ,
            label_value=l2_label, label_status=l2_status,
            predicted=l2_pred, gates=l2_gates,
        )

        cells_out.append({
            "slug": slug,
            "shape_point": shape_point,
            "binary_sha256": key[2],
            "kernel_signature_sha256": key[3],
            "primitives": [read_row, write_row, l2_row],
        })

    # ---- aggregates ----
    per_primitive: dict[str, list[float]] = {
        "read_sectors": [], "write_sectors": [], "l2_requests": [],
    }
    scored_slugs: set[str] = set()
    status_counts: dict[str, int] = {}
    for cell in cells_out:
        for row in cell["primitives"]:
            status_counts[row["status"]] = status_counts.get(
                row["status"], 0) + 1
            if row["status"] == "scored":
                per_primitive[row["primitive"]].append(
                    row["rel_error"])
                scored_slugs.add(cell["slug"])
    pooled = [r for rels in per_primitive.values() for r in rels]
    floor_cfg = PREDICTOR_CONFIG["non_vacuity_floor"]
    floor_met = (
        len(pooled) >= int(floor_cfg["min_scored_pairs"])
        and len(scored_slugs) >= int(floor_cfg["min_distinct_slugs"])
    )
    targets = PREDICTOR_CONFIG["directional_targets"]

    def _avt(name: str, stats: dict) -> list[dict]:
        rows = []
        for q, target in (("p50", targets["p50"]),
                          ("p90", targets["p90"])):
            achieved = stats[q]
            rows.append({
                "gate": f"{name}_{q}_rel_error",
                "target": target,
                "achieved": achieved,
                "met": (
                    None if (achieved is None or not floor_met)
                    else achieved <= target
                ),
            })
        return rows

    achieved_vs_target: list[dict] = []
    achieved_vs_target += _avt("pooled", _stats(pooled))
    for name, rels in per_primitive.items():
        achieved_vs_target += _avt(name, _stats(rels))

    # ---- hit-rate pp-bucket report (from COMMITTED rescore pairs;
    # no label re-read, no holdout consumption) ----
    barred_by_cell = {
        (c["slug"], c["shape_point"]):
            next(r for r in c["primitives"]
                 if r["primitive"] == "read_sectors")["status"]
        for c in cells_out
    }
    buckets = PREDICTOR_CONFIG["hit_rate_buckets"]
    pp_rows: list[dict] = []
    for pair in (rescore["pool"]["pairs"]
                 + rescore["holdout"]["pairs"]):
        if pair["status"] != "scored":
            continue
        label = pair["label_hit_rate"]
        bucket = next(
            (b for b, (lo, hi) in buckets.items()
             if lo <= label < hi), "high")
        read_status = barred_by_cell.get(
            (pair["slug"], pair["shape_point"]))
        promoted = read_status == "scored"
        pp_rows.append({
            "slug": pair["slug"],
            "shape_point": pair["shape_point"],
            "split": pair["split"],
            "abs_error_pp": pair["abs_error_pp"],
            "label_hit_rate": label,
            "bucket": bucket,
            "promoted": promoted,
            "promotion_bar_reason": (
                None if promoted else
                f"read_sectors_primitive_{read_status}"
            ),
        })
    pp_targets = PREDICTOR_CONFIG["hit_rate_pp_targets"]
    promoted_pp = sorted(
        r["abs_error_pp"] for r in pp_rows if r["promoted"])
    bucket_stats = {}
    for bucket in buckets:
        vals = sorted(
            r["abs_error_pp"] for r in pp_rows
            if r["promoted"] and r["bucket"] == bucket)
        bucket_stats[bucket] = {
            "n": len(vals),
            "p50_pp": _pctile(vals, 0.50) if vals else None,
            "p90_pp": _pctile(vals, 0.90) if vals else None,
        }
    hit_rate_report = {
        "source": str(rescore_path.relative_to(_REPO_ROOT)),
        "note": (
            "computed from the committed re-score pairs (the single "
            "holdout touch's as-generated output); no label re-read, "
            "no governance budget consumed"
        ),
        "rows": sorted(
            pp_rows,
            key=lambda r: (r["slug"], r["shape_point"], r["split"])),
        "n_promoted": len(promoted_pp),
        "n_barred": len(pp_rows) - len(promoted_pp),
        "overall_promoted": {
            "p50_pp": (
                _pctile(promoted_pp, 0.50) if promoted_pp else None),
            "p90_pp": (
                _pctile(promoted_pp, 0.90) if promoted_pp else None),
        },
        "buckets": bucket_stats,
        "achieved_vs_target": [
            {
                "gate": "hit_rate_p50_abs_pp",
                "target": pp_targets["p50_pp"],
                "achieved": (
                    _pctile(promoted_pp, 0.50)
                    if promoted_pp else None),
                "met": (
                    _pctile(promoted_pp, 0.50)
                    <= pp_targets["p50_pp"]
                    if promoted_pp else None),
            },
            {
                "gate": "hit_rate_p90_abs_pp",
                "target": pp_targets["p90_pp"],
                "achieved": (
                    _pctile(promoted_pp, 0.90)
                    if promoted_pp else None),
                "met": (
                    _pctile(promoted_pp, 0.90)
                    <= pp_targets["p90_pp"]
                    if promoted_pp else None),
            },
        ],
    }

    return {
        "schema_version": 1,
        "artifact_kind": "t2_traffic_prediction",
        "arch": "sm_86",
        "generated_at_commit": _git_head(),
        "config": PREDICTOR_CONFIG,
        "config_digest": config_digest(),
        "inputs": {
            "labels": str(labels_path.relative_to(_REPO_ROOT)),
            "t2_exclusions": str(
                t2_exclusions_path.relative_to(_REPO_ROOT)),
            "replay_emissions": str(
                emissions_path.relative_to(_REPO_ROOT)),
            "walker_artifact": str(
                walker_path.relative_to(_REPO_ROOT)),
            "rescore_artifact": str(
                rescore_path.relative_to(_REPO_ROOT)),
        },
        "n_cells": len(cells_out),
        "status_counts": status_counts,
        "non_vacuity_floor": {
            **floor_cfg,
            "scored_pairs": len(pooled),
            "distinct_scored_slugs": len(scored_slugs),
            "met": floor_met,
        },
        "achieved_vs_target": achieved_vs_target,
        "cells": cells_out,
        "hit_rate_pp_report": hit_rate_report,
    }


def _validate(artifact: dict, *, schema_path: Path) -> None:
    import jsonschema
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(artifact, schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--t2-exclusions", type=Path, default=DEFAULT_T2_EXCLUSIONS)
    parser.add_argument(
        "--emissions", type=Path, default=DEFAULT_EMISSIONS)
    parser.add_argument(
        "--walker", type=Path, default=DEFAULT_WALKER)
    parser.add_argument(
        "--rescore", type=Path, default=DEFAULT_RESCORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--schema-path", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    try:
        artifact = build_prediction(
            labels_path=args.labels,
            t2_exclusions_path=args.t2_exclusions,
            emissions_path=args.emissions,
            walker_path=args.walker,
            rescore_path=args.rescore,
        )
    except RescoreGovernanceError as exc:
        sys.stderr.write(f"governance failure: {exc}\n")
        return 2
    _validate(artifact, schema_path=args.schema_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"t2_traffic_predictor: wrote {args.out} "
        f"({artifact['n_cells']} cells; "
        f"statuses {artifact['status_counts']}; floor_met="
        f"{artifact['non_vacuity_floor']['met']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
