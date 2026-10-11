#!/usr/bin/env python3
"""Logical SM120 cache/service envelope driven by direct trace features.

This is an executable finite-service model, not a recovered RTX 5090 cache
geometry.  It consumes direct trace demand and explicit scenario parameters;
NCU rows may be joined only for a separately labelled diagnostic table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def ncu_sector_observation(row: dict[str, Any]) -> dict[str, float]:
    # Replay rows carry the independent profile under ``ncu_metrics``;
    # accepting ``metrics`` as well keeps this adapter compatible with compact
    # profile rows produced by older campaigns.
    metrics = row.get("ncu_metrics") or row.get("metrics") or {}
    out: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(value, dict) or "sector" not in str(value.get("unit", "")):
            continue
        if any(token in name for token in ("l1tex__t_sectors", "lts__t_sectors", "dram__sectors")):
            try:
                out[name] = float(value["value"])
            except (TypeError, ValueError):
                continue
    return out


def predict_launch(
    launch: dict[str, Any],
    config: dict[str, Any],
    mrc_scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capacities = [int(x) for x in config.get("l1_capacity_sectors", [128, 256, 512, 1024])]
    curve = launch.get("locality", {}).get("capacity_sectors", {})
    # The old direct-locality surface counts lane address records while its
    # cache curve counts 32-byte sectors.  Prefer the MRC's modeled-sector
    # denominator when present so service demand and miss ratios share a unit.
    mrc_curve = (mrc_scenario or {}).get("mrc", {})
    mrc_used = bool(mrc_scenario and int(mrc_scenario.get("references", 0) or 0) > 0)
    if mrc_used:
        demand = int(mrc_scenario.get("references", 0) or 0)
        demand_basis = "mrc_modeled_sector_references"
    else:
        demand = max(0, int(launch.get("address_records", 0)))
        demand_basis = "trace_address_records_legacy_fallback"
    tma_bytes = max(0, int((launch.get("tma") or {}).get("bytes", 0)))
    tma_sectors = math.ceil(tma_bytes / int(config.get("sector_bytes", 32)))
    records: list[dict[str, Any]] = []
    for capacity in capacities:
        if mrc_used and str(capacity) in mrc_curve:
            l1 = mrc_curve[str(capacity)]
            hits = min(demand, max(0, int(l1.get("hits", 0) or 0)))
            misses = min(demand, max(0, int(l1.get("misses", 0) or 0)))
            mrc_source = "mrc_scenario"
        else:
            l1 = curve.get(str(capacity), {})
            hits = int(l1.get("hits", 0) or 0)
            misses = int(l1.get("misses", max(0, demand - hits)) or 0)
            mrc_source = "legacy_trace_locality_curve"
        l2_hits = min(misses, int(config.get("logical_l2_resident_fraction", 0.5) * misses))
        dram_misses = max(0, misses - l2_hits)
        l1_cycles = hits * int(config.get("l1_hit_cycles", 20))
        l2_cycles = l2_hits * int(config.get("l2_hit_cycles", 100))
        dram_cycles = dram_misses * int(config.get("dram_service_cycles", 300))
        total_demand = misses + tma_sectors
        queue_capacity = max(1, int(config.get("outstanding_capacity", 128)))
        batches = math.ceil(total_demand / queue_capacity) if total_demand else 0
        queue_cycles = batches * int(config.get("service_batch_cycles", 32))
        records.append(
            {
                "capacity_sectors": capacity,
                "trace_demand_sectors": demand,
                "demand_basis": demand_basis,
                "tma_sectors": tma_sectors,
                "l1_hits": hits,
                "l1_misses": misses,
                "logical_l2_hits": l2_hits,
                "logical_dram_misses": dram_misses,
                "queue_batches": batches,
                "predicted_service_cycles": l1_cycles + l2_cycles + dram_cycles + queue_cycles,
                "scenario": "fully_associative_LRU_plus_finite_logical_service",
                "mrc_source": mrc_source,
                "mrc_sample_bin_references": int((mrc_scenario or {}).get("sample_bin_references", 0) or 0),
            }
        )
    return {
        "kernel_id": launch.get("kernel_id"),
        "records": records,
        "mrc_scenario": (mrc_scenario or {}).get("scenario_semantics") if mrc_scenario else None,
        "mrc_available": mrc_used,
    }


def load_mrc_index(path: Path | None, scenario_name: str) -> dict[tuple[str, int], dict[str, Any]]:
    """Index MRC rows by capture and launch ordinal without using NCU data."""
    if not path or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for capture in payload.get("captures", []):
        capture_name = str(capture.get("capture_name"))
        for launch_index, launch in enumerate(capture.get("launches", [])):
            scenario = (launch.get("reuse_distance_scenarios") or {}).get(scenario_name)
            if isinstance(scenario, dict):
                index[(capture_name, launch_index)] = scenario
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument(
        "--mrc",
        type=Path,
        help="optional sm120_mrc_model JSON; used only as a logical scenario input",
    )
    parser.add_argument(
        "--mrc-scenario",
        default="instrumented_receive_order",
        choices=["instrumented_receive_order", "reverse_warp_cta_order", "warm_second_pass"],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = {
        "sector_bytes": 32,
        "l1_capacity_sectors": [128, 256, 512, 1024, 2048],
        "logical_l2_resident_fraction": 0.5,
        "l1_hit_cycles": 20,
        "l2_hit_cycles": 100,
        "dram_service_cycles": 300,
        "outstanding_capacity": 128,
        "service_batch_cycles": 32,
    }
    if args.config and args.config.is_file():
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    features = json.loads(args.features.read_text(encoding="utf-8"))
    mrc_index = load_mrc_index(args.mrc, args.mrc_scenario)
    ncu_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if args.replay and args.replay.is_file():
        for line in args.replay.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for item in (record.get("simulator") or {}).get("per_launch", []) or []:
                ncu_rows[(str(record.get("capture_name")), int(item.get("launch_index", -1)))] = item
    outputs = []
    for capture in features.get("captures", []):
        for index, launch in enumerate(capture.get("launches", [])):
            capture_name = str(capture.get("capture_name"))
            mrc_scenario = mrc_index.get((capture_name, index))
            pred = predict_launch(launch, config, mrc_scenario)
            ncu = ncu_rows.get((str(capture.get("capture_name")), index))
            pred["capture_name"] = capture_name
            pred["launch_index"] = index
            pred["ncu_diagnostic"] = {
                "status": "separate_observation_join" if ncu else "unavailable",
                "metrics": ncu_sector_observation(ncu) if ncu else {},
                "prediction_inputs": "none",
            }
            outputs.append(pred)
    result = {
        "schema_version": 1,
        "model": "sm120_logical_cache_service_v1",
        "configuration": config,
        "mrc_input": str(args.mrc) if args.mrc else None,
        "mrc_scenario": args.mrc_scenario if args.mrc else None,
        "prediction_rows": outputs,
        "physical_non_claim": ["set_hash", "associativity", "MSHR_count", "DRAM_arbitration_policy"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(outputs), "model": result["model"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
