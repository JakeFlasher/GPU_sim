#!/usr/bin/env python3
"""Compare two post-replay DRAM-read demand projections.

The legacy projection adds TMA payload sectors to the ordinary replay DRAM
read counter.  The alternative replaces those payload sectors with misses
from the address-derived TMA source shadow.  This is strictly a retrospective
comparison against independent NCU rows: neither NCU values nor the shadow
projection is consumed by replay, and the shadow does not mutate cache or
timing state.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from jsonl_io import open_jsonl_reader
from sm120_replay_bench import join_ncu_rows, parse_ncu_raw


def finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def family(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered) if ordered else None,
        "p90": ordered[int(0.90 * (len(ordered) - 1))] if ordered else None,
    }


def new_bucket() -> dict[str, object]:
    return {
        "rows": 0,
        "shadow_available_rows": 0,
        "shadow_nonzero_rows": 0,
        "legacy_errors": [],
        "shadow_errors": [],
        "available_legacy_errors": [],
        "available_shadow_errors": [],
        "nonzero_legacy_errors": [],
        "nonzero_shadow_errors": [],
        "better": 0,
        "worse": 0,
        "available_better": 0,
        "available_worse": 0,
        "nonzero_better": 0,
        "nonzero_worse": 0,
    }


def add(bucket: dict[str, object], legacy: float, shadow: float, available: bool, nonzero: bool) -> None:
    bucket["rows"] = int(bucket["rows"]) + 1
    bucket["legacy_errors"].append(legacy)
    bucket["shadow_errors"].append(shadow)
    if shadow < legacy:
        bucket["better"] = int(bucket["better"]) + 1
    elif shadow > legacy:
        bucket["worse"] = int(bucket["worse"]) + 1
    if available:
        bucket["shadow_available_rows"] = int(bucket["shadow_available_rows"]) + 1
        bucket["available_legacy_errors"].append(legacy)
        bucket["available_shadow_errors"].append(shadow)
        if shadow < legacy:
            bucket["available_better"] = int(bucket["available_better"]) + 1
        elif shadow > legacy:
            bucket["available_worse"] = int(bucket["available_worse"]) + 1
    if nonzero:
        bucket["shadow_nonzero_rows"] = int(bucket["shadow_nonzero_rows"]) + 1
        bucket["nonzero_legacy_errors"].append(legacy)
        bucket["nonzero_shadow_errors"].append(shadow)
        if shadow < legacy:
            bucket["nonzero_better"] = int(bucket["nonzero_better"]) + 1
        elif shadow > legacy:
            bucket["nonzero_worse"] = int(bucket["nonzero_worse"]) + 1


def render_bucket(bucket: dict[str, object]) -> dict[str, object]:
    return {
        "rows": bucket["rows"],
        "shadow_available_rows": bucket["shadow_available_rows"],
        "shadow_nonzero_rows": bucket["shadow_nonzero_rows"],
        "legacy_error": distribution(bucket["legacy_errors"]),
        "shadow_error": distribution(bucket["shadow_errors"]),
        "available_legacy_error": distribution(bucket["available_legacy_errors"]),
        "available_shadow_error": distribution(bucket["available_shadow_errors"]),
        "nonzero_legacy_error": distribution(bucket["nonzero_legacy_errors"]),
        "nonzero_shadow_error": distribution(bucket["nonzero_shadow_errors"]),
        "better": bucket["better"],
        "worse": bucket["worse"],
        "available_better": bucket["available_better"],
        "available_worse": bucket["available_worse"],
        "nonzero_better": bucket["nonzero_better"],
        "nonzero_worse": bucket["nonzero_worse"],
    }


def summarize(full: Path, profiles: Path) -> dict:
    overall = new_bucket()
    families: dict[str, dict[str, object]] = defaultdict(new_bucket)
    capture_cache: dict[str, list[dict]] = {}
    captures = 0
    launches = 0
    with open_jsonl_reader(full) as stream:
        for line in stream:
            capture = json.loads(line)
            name = str(capture.get("canonical_capture", ""))
            if name not in capture_cache:
                capture_cache[name] = parse_ncu_raw(profiles / name / "raw.csv")
            ncu_rows = capture_cache[name]
            launches += len(capture.get("launches", []))
            captures += 1
            fam = families[family(name)]
            for joined in join_ncu_rows(capture.get("launches", []), ncu_rows):
                li = joined.get("launch_index")
                ni = joined.get("ncu_row_index")
                if li is None or ni is None:
                    continue
                launch = capture["launches"][li]
                target_entry = (ncu_rows[ni].get("metrics") or {}).get(
                    "dram__sectors_op_read.sum"
                )
                target = target_entry.get("value") if isinstance(target_entry, dict) else None
                if not finite(target):
                    continue
                counters = launch.get("counters") or {}
                dram = counters.get("dram") or {}
                front_end = counters.get("front_end") or {}
                trace = launch.get("trace_metrics") or {}
                ordinary = int(dram.get("read_sectors", 0))
                compulsory = int(front_end.get("compulsory_instruction_read_sectors", 0))
                payload = int(trace.get("tma_sidecar_load_payload_bytes", 0)) // 32
                shadow_misses = int(trace.get("tma_shadow_l2_misses", 0))
                shadow_requests = int(trace.get("tma_shadow_l2_requests", 0))
                matched = int(trace.get("tma_shadow_dynamic_records_matched", 0))
                unavailable = int(trace.get("tma_shadow_l2_address_unavailable", 0))
                legacy = abs(ordinary + payload + compulsory - float(target))
                shadow = abs(ordinary + shadow_misses + compulsory - float(target))
                available = bool(shadow_requests or matched or shadow_misses or unavailable)
                nonzero = bool(payload or shadow_misses)
                add(overall, legacy, shadow, available, nonzero)
                add(fam, legacy, shadow, available, nonzero)
    return {
        "scope": (
            "retrospective DRAM-read demand comparison; address-derived TMA shadow "
            "is diagnostic/conditional; NCU is independent post-replay evidence"
        ),
        "source": str(full),
        "profiles": str(profiles),
        "captures": captures,
        "launches": launches,
        "overall": render_bucket(overall),
        "families": {name: render_bucket(value) for name, value in sorted(families.items())},
        "decision": (
            "retain_shadow_diagnostic_only: sidecar source order is not established "
            "relative to ordinary dynamic L2 requests; no replay timing/cache state changed"
        ),
    }


def markdown(result: dict) -> str:
    overall = result["overall"]
    lines = [
        "# TMA source-shadow DRAM-read demand comparison",
        "",
        "Retrospective comparison only. NCU is an independent post-replay target;",
        "neither projection is fed into replay and the source shadow does not mutate",
        "the shared L2 or timing state.",
        "",
        f"- Captures: {result['captures']}",
        f"- Launches: {result['launches']}",
        f"- Compared rows: {overall['rows']}",
        f"- Shadow-available rows: {overall['shadow_available_rows']}",
        f"- Rows improved / worsened: {overall['better']} / {overall['worse']}",
        f"- Legacy absolute-error median/P90: {overall['legacy_error']['median']} / {overall['legacy_error']['p90']} sectors",
        f"- Shadow absolute-error median/P90: {overall['shadow_error']['median']} / {overall['shadow_error']['p90']} sectors",
        "",
        "| family | rows | shadow available | legacy median/P90 | shadow median/P90 | better/worse |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["families"].items():
        legacy = value["legacy_error"]
        shadow = value["shadow_error"]
        lines.append(
            f"| {name} | {value['rows']} | {value['shadow_available_rows']} | "
            f"{legacy['median']} / {legacy['p90']} | {shadow['median']} / {shadow['p90']} | "
            f"{value['better']} / {value['worse']} |"
        )
    lines.extend(
        [
            "",
            "The shadow changes only rows with source-address evidence. Its",
            "overall effect is small and family-dependent (notably improving",
            "Cutlass while worsening quant/SASS rows in the available subset).",
            "Because sidecar source order is not causally interleaved with ordinary",
            "dynamic L2 requests, the shadow is retained as a conditional diagnostic",
            "and is not promoted into the default DRAM or L2 replay path.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.full, args.profiles)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
