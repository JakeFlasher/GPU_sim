#!/usr/bin/env python3
"""Audit whether captured driver CTA limits can change traced SM admission.

This is an NCU-independent post-replay structural check. It compares each
launch's recorded driver occupancy limit with the maximum number of live traced
warps on any SM and the launch block's warp cardinality. It never changes or
selects simulator parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonl_io import open_jsonl_reader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {
        "full_replay": str(args.full),
        "captures": 0,
        "launches": 0,
        "driver_limited_launches": 0,
        "observed_peak_exceeds_driver": 0,
        "observed_peak_warp_exceeds_driver": 0,
        "invalid_zero_driver_limits": 0,
        "examples": [],
    }
    with open_jsonl_reader(args.full) as stream:
        for line in stream:
            capture = json.loads(line)
            summary["captures"] += 1
            for launch_index, launch in enumerate(capture.get("launches", [])):
                summary["launches"] += 1
                resources = (launch.get("trace_metrics") or {}).get("launch_resources") or {}
                limit = resources.get("driver_occupancy_limit_blocks")
                block_threads = resources.get("block_threads") or 0
                if not isinstance(limit, int) or limit == 0:
                    summary["invalid_zero_driver_limits"] += 1
                    continue
                summary["driver_limited_launches"] += 1
                warp_per_block = (block_threads + 31) // 32
                peak_warps = max(
                    (
                        activity.get("peak_live_warps", 0)
                        for activity in (launch.get("sm_activity") or {}).values()
                    ),
                    default=0,
                )
                peak_blocks = peak_warps // warp_per_block if warp_per_block else 0
                if peak_blocks > limit:
                    summary["observed_peak_exceeds_driver"] += 1
                    if len(summary["examples"]) < 20:
                        summary["examples"].append(
                            {
                                "capture": capture.get("canonical_capture"),
                                "launch_index": launch_index,
                                "peak_live_warps": peak_warps,
                                "warp_per_block": warp_per_block,
                                "driver_limit": limit,
                            }
                        )
                if peak_warps > limit * warp_per_block:
                    summary["observed_peak_warp_exceeds_driver"] += 1
    summary["no_op"] = summary["observed_peak_warp_exceeds_driver"] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
