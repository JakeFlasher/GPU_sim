#!/usr/bin/env python3
"""Reduce the prospective SM86 cross-warp async queue-overlap smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = (
    REPO_ROOT / "results/captures/sm86_async_cross_warp_queue_overlap_20260831_v1"
)
DEFAULT_AGGREGATE = (
    REPO_ROOT / "data/local_labels/sm86_async_cross_warp_queue_overlap_report.json"
)
ROUTES = ("l1", "l2")
ARMS = ("queued", "drained")
EXACT_NCU = {
    "smsp__inst_executed_op_ldgsts.sum": 2048.0,
    "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts.sum": 32768.0,
    "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum": 8192.0,
}


class AnalysisError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{path}: expected one JSON object")
    return value


def require_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{context}: expected object")
    return value


def number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{context}: expected number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise AnalysisError(f"{context}: expected finite number")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_static(root: Path) -> dict[str, Any]:
    summary = read_json(root / "static/summary.json")
    if summary.get("target_function_count") != 4:
        raise AnalysisError("static target function count differs")
    if summary.get("static_shared_bytes") != 33112:
        raise AnalysisError("static shared footprint differs")
    control = read_json(root / "static/control_bits.json")
    control_summary = require_object(control.get("summary"), "control summary")
    if control.get("arch") != "sm_86" or control_summary.get(
        "total_unknown_encoding"
    ) != 0:
        raise AnalysisError("control sidecar is not a complete sm_86 decode")
    functions = control.get("functions")
    if not isinstance(functions, Mapping) or len(functions) != 5:
        raise AnalysisError("control sidecar must contain five functions")
    return {
        "passed": True,
        "binary": summary.get("binary"),
        "source": summary.get("source"),
        "static_shared_bytes": 33112,
        "control_sha256": sha256(root / "static/control_bits.json"),
        "decoded_instructions": control_summary.get("total_decoded"),
    }


def validate_direct(root: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for route in ROUTES:
        path = root / "direct/smoke/repeat-1" / f"{route}.json"
        value = read_json(path)
        expected = {
            "workload": "async_cross_warp_queue_overlap",
            "mode": "paired",
            "route": route,
            "active_warps": 4,
            "groups": 64,
            "copies_per_group": 8,
            "background_groups_per_epoch": 4,
            "correct": True,
            "result_store_after_intervals": True,
        }
        for field, wanted in expected.items():
            if value.get(field) != wanted:
                raise AnalysisError(f"{path}: {field} differs")
        queued = require_object(value.get("queued"), f"{path}: queued")
        drained = require_object(value.get("drained"), f"{path}: drained")
        for name, arm in (("queued", queued), ("drained", drained)):
            issue = number(arm.get("issue_cycles"), f"{path}: {name} issue")
            wait = number(arm.get("wait_cycles"), f"{path}: {name} wait")
            total = number(arm.get("total_cycles"), f"{path}: {name} total")
            if issue + wait != total:
                raise AnalysisError(f"{path}: {name} interval sum differs")
            if arm.get("commit_before_target_sum") != 192:
                raise AnalysisError(f"{path}: {name} commit ordering differs")
        if queued.get("overlap_background_sum") != 192:
            raise AnalysisError(f"{path}: queued overlap population differs")
        if queued.get("overlap_threshold_epochs") != 64:
            raise AnalysisError(f"{path}: queued overlap threshold failed")
        if queued.get("zero_overlap_epochs") != 0:
            raise AnalysisError(f"{path}: queued contains a zero-overlap epoch")
        if drained.get("overlap_background_sum") != 0:
            raise AnalysisError(f"{path}: drained contains overlap")
        if drained.get("zero_overlap_epochs") != 64:
            raise AnalysisError(f"{path}: drained zero-overlap count differs")
        cells[route] = {
            "order": value.get("order"),
            "issue_delta_per_group":
                (number(queued.get("issue_cycles"), "queued issue")
                 - number(drained.get("issue_cycles"), "drained issue"))
                / 64.0,
            "wait_delta_per_group":
                (number(queued.get("wait_cycles"), "queued wait")
                 - number(drained.get("wait_cycles"), "drained wait"))
                / 64.0,
            "queued_overlap_backgrounds": queued.get("overlap_background_sum"),
            "drained_overlap_backgrounds": drained.get("overlap_background_sum"),
        }
    return {"passed": True, "manipulation_passed": True, "cells": cells}


def validate_nvbit(root: Path, binary_sha: str) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for route in ROUTES:
        for arm in ARMS:
            key = f"{route}_{arm}_w4_g64"
            cell = root / "nvbit" / key
            complete = read_json(cell / "complete.json")
            binding = read_json(cell / "campaign_binding.json")
            if complete.get("complete") is not True:
                raise AnalysisError(f"{key}: incomplete NVBit capture")
            if binding.get("binary", {}).get("sha256") != binary_sha:
                raise AnalysisError(f"{key}: NVBit binary identity differs")
            with (cell / "stats.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise AnalysisError(f"{key}: expected one stats row")
            row = {str(k).strip(): str(v).strip() for k, v in rows[0].items()}
            total = int(row["total_insts"])
            reported = int(row["total_reported_insts"])
            if total <= 0 or total != reported:
                raise AnalysisError(f"{key}: incomplete dynamic instruction stream")
            if "queue_overlap_endpoint_kernel" not in row["kernel mangled name"]:
                raise AnalysisError(f"{key}: wrong traced kernel")
            cells[key] = {
                "instructions": total,
                "trace_sha256": sha256(cell / "dynamic_trace.pb"),
                "payload_files": binding.get("output_contract", {}).get(
                    "payload_file_count"
                ),
            }
    return {"passed": True, "trace_count": len(cells), "cells": cells}


def validate_ncu(root: Path, binary_sha: str) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    route_metrics: dict[str, dict[str, Mapping[str, Any]]] = {}
    for route in ROUTES:
        route_metrics[route] = {}
        for arm in ARMS:
            key = f"{route}_{arm}_w4_g64"
            path = root / "ncu" / key / "summary.json"
            summary = read_json(path)
            if summary.get("route") != route or summary.get("arm") != arm:
                raise AnalysisError(f"{key}: NCU identity differs")
            if summary.get("binary", {}).get("sha256") != binary_sha:
                raise AnalysisError(f"{key}: NCU binary identity differs")
            metrics = require_object(summary.get("metrics"), f"{key}: metrics")
            for metric, expected in EXACT_NCU.items():
                if number(metrics.get(metric), f"{key}: {metric}") != expected:
                    raise AnalysisError(f"{key}: exact NCU work differs")
            route_metrics[route][arm] = metrics
            cells[key] = {
                "metrics": dict(metrics),
                "report_sha256": summary.get("report", {}).get("sha256"),
            }
        queued = route_metrics[route]["queued"]
        drained = route_metrics[route]["drained"]
        if dict(queued) != dict(drained):
            raise AnalysisError(f"{route}: queued/drained NCU populations differ")

    l1 = route_metrics["l1"]["queued"]
    l1_hits = number(
        l1.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum"),
        "L1 hits",
    )
    l1_misses = number(
        l1.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"),
        "L1 misses",
    )
    l1_ratio = l1_hits / (l1_hits + l1_misses)
    l2 = route_metrics["l2"]["queued"]
    l2_l1_misses = number(
        l2.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"),
        "L2 route L1 misses",
    )
    l2_total = number(
        l2.get("lts__t_sectors_srcunit_tex_aperture_device_op_read.sum"),
        "L2 total",
    )
    l2_hits = number(
        l2.get("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum"),
        "L2 hits",
    )
    if l1_ratio < 0.95 or l2_l1_misses / 32768.0 < 0.95 or l2_hits / l2_total < 0.95:
        raise AnalysisError("NCU route dominance gate failed")
    return {
        "passed": True,
        "paired_equal": True,
        "l1_hit_ratio": l1_ratio,
        "l2_l1_miss_ratio": l2_l1_misses / 32768.0,
        "l2_lts_hit_ratio": l2_hits / l2_total,
        "cells": cells,
    }


def replay_launch(path: Path) -> Mapping[str, Any]:
    value = read_json(path)
    launches = value.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise AnalysisError(f"{path}: expected one launch")
    return require_object(launches[0], f"{path}: launch")


def validate_replays(root: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for route in ROUTES:
        for arm in ARMS:
            key = f"{route}_{arm}_w4_g64"
            directory = root / "simulator" / key
            first = directory / "replay-1.json"
            second = directory / "replay-2.json"
            first_sha = sha256(first)
            if first_sha != sha256(second):
                raise AnalysisError(f"{key}: replay twins differ")
            launch = replay_launch(first)
            timing = require_object(launch.get("timing"), f"{key}: timing")
            readiness = require_object(timing.get("readiness"), f"{key}: readiness")
            if readiness.get("schema_version") != 20:
                raise AnalysisError(f"{key}: readiness schema is not 20")
            overlap = require_object(
                readiness.get("async_issue_overlap"), f"{key}: overlap"
            )
            expected_route_copies = 1536
            if overlap.get("coissue_cycles") != 512:
                raise AnalysisError(f"{key}: coissue cycle count differs")
            if overlap.get("coissued_copies") != expected_route_copies:
                raise AnalysisError(f"{key}: coissued copy count differs")
            if overlap.get("single_acceptor_excess_copies") != 1024:
                raise AnalysisError(f"{key}: Q1 excess differs")
            if overlap.get("maximum_coissue_width") != 3:
                raise AnalysisError(f"{key}: maximum coissue width differs")
            route_field = (
                "cache_all_coissued_copies" if route == "l1" else "bypass_coissued_copies"
            )
            if overlap.get(route_field) != expected_route_copies:
                raise AnalysisError(f"{key}: route-split coissue differs")
            frontier = require_object(
                timing.get("completion_frontier"), f"{key}: completion frontier"
            )
            tails = require_object(
                frontier.get("unmodeled_tails"), f"{key}: tails"
            )
            if tails.get("async_copy_shared_visibility") is not (route == "l2"):
                raise AnalysisError(f"{key}: async visibility tail differs")
            groups = require_object(readiness.get("async_groups"), f"{key}: groups")
            cells[key] = {
                "replay_sha256": first_sha,
                "diagnostic_last_issue_cycles": timing.get(
                    "diagnostic_last_issue_cycles"
                ),
                "async_issue_overlap": dict(overlap),
                "async_groups": dict(groups),
                "typed_tails": dict(tails),
            }
        queued = cells[f"{route}_queued_w4_g64"]
        drained = cells[f"{route}_drained_w4_g64"]
        if queued["async_issue_overlap"] != drained["async_issue_overlap"]:
            raise AnalysisError(f"{route}: modeled overlap populations differ")
    predictions = {
        route: (
            number(
                cells[f"{route}_queued_w4_g64"]["diagnostic_last_issue_cycles"],
                f"{route} queued frontier",
            )
            - number(
                cells[f"{route}_drained_w4_g64"]["diagnostic_last_issue_cycles"],
                f"{route} drained frontier",
            )
        )
        / 64.0
        for route in ROUTES
    }
    return {
        "passed": True,
        "deterministic": True,
        "cells": cells,
        "wait_predictions_per_group": predictions,
        "cache_all_prediction_material": abs(predictions["l1"]) >= 16.0,
    }


def require_absent(root: Path) -> dict[str, Any]:
    paths = [
        root / "direct/identification",
        root / "direct/holdout",
        root / "nvbit/holdout",
        root / "ncu/holdout",
        root / "simulator/holdout",
    ]
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise AnalysisError(f"unauthorized full/holdout artifacts exist: {present}")
    return {
        "identification_authorized": False,
        "identification_captured": False,
        "identification_timing_read": False,
        "holdout_authorized": False,
        "holdout_captured": False,
        "holdout_timing_read": False,
    }


def build_report(root: Path) -> dict[str, Any]:
    static = validate_static(root)
    binary = require_object(static.get("binary"), "static binary")
    binary_sha = str(binary.get("sha256"))
    direct = validate_direct(root)
    nvbit = validate_nvbit(root, binary_sha)
    ncu = validate_ncu(root, binary_sha)
    replays = validate_replays(root)
    unopened = require_absent(root)
    if replays["cache_all_prediction_material"]:
        raise AnalysisError("cache-all control became material but full identification is absent")
    prediction = replays["wait_predictions_per_group"]
    return {
        "schema_version": 1,
        "assay": "sm86_async_cross_warp_queue_overlap_v1",
        "status": "complete_cross_warp_queue_control_no_go",
        "decision": (
            "Exact queued/drained live manipulation and route/work gates pass, but "
            "per-warp traces reconstruct identical schema-20 overlap populations and "
            "the zero-fit cache-all prediction is non-material; full labels and W8 "
            "holdout remain unopened."
        ),
        "root": str(root.resolve()),
        "timing_promotion_authorized": False,
        "typed_visibility_tail_required": True,
        "static": static,
        "direct_smoke": direct,
        "nvbit": nvbit,
        "ncu": ncu,
        "simulator_replays": replays,
        "unopened": unopened,
        "counts": {
            "direct_smoke_runs": 2,
            "ncu_reports": 4,
            "nvbit_traces": 4,
            "simulator_outputs": 8,
            "full_identification_runs": 0,
            "holdout_runs": 0,
        },
        "criteria": {
            "structural_smoke_passed": True,
            "live_manipulation_passed": True,
            "route_admission_passed": True,
            "cache_all_prediction_cycles_per_group": prediction["l1"],
            "cache_all_prediction_material": False,
            "identification_authorized": False,
            "holdout_untouched": True,
            "timing_unchanged": True,
            "schema20_timing_neutral": True,
        },
    }


def markdown(report: Mapping[str, Any]) -> str:
    direct = require_object(report["direct_smoke"], "direct")
    cells = require_object(direct["cells"], "direct cells")
    replay = require_object(report["simulator_replays"], "replay")
    prediction = require_object(replay["wait_predictions_per_group"], "prediction")
    return f"""# SM86 asynchronous-copy cross-warp queue-overlap result

Status: `{report['status']}`

Queued/drained source, destination, route, and work identities pass. Every
queued epoch has all three background intervals overlapping the target; every
drained epoch has zero overlap. NCU and NVBit structural evidence is exact.

| Route | Live smoke issue delta | Live smoke wait delta | Simulator frontier delta |
|---|---:|---:|---:|
| cache-all L1 | {cells['l1']['issue_delta_per_group']:.6f} | {cells['l1']['wait_delta_per_group']:.6f} | {prediction['l1']:.6f} |
| L2-hit bypass | {cells['l2']['issue_delta_per_group']:.6f} | {cells['l2']['wait_delta_per_group']:.6f} | {prediction['l2']:.6f} |

The cache-all simulator control is below the prospectively required 16-cycle
magnitude. Queued and drained endpoint traces reconstruct identical schema-20
overlap populations because the retained trace format has no cross-warp service
order/timestamps. The ten-pair identification and W8 holdout were not captured
or read. No timing or typed-tail state changes.
"""


def write_or_validate(path: Path, payload: bytes, validate_only: bool) -> None:
    if validate_only:
        if path.read_bytes() != payload:
            raise AnalysisError(f"{path}: retained output differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--aggregate-output", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / "analysis/report.json"
    output_md = args.output_md or root / "analysis/report.md"
    try:
        report = build_report(root)
        encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        rendered = markdown(report).encode()
        write_or_validate(output, encoded, args.validate_only)
        write_or_validate(output_md, rendered, args.validate_only)
        write_or_validate(args.aggregate_output, encoded, args.validate_only)
    except (AnalysisError, OSError, ValueError, KeyError) as error:
        print(f"cross-warp queue analysis: {error}", file=sys.stderr)
        return 1
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
