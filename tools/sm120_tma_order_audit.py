#!/usr/bin/env python3
"""Report whether captured TMA rows have a shared dynamic ordering anchor.

This is a trace-evidence report only.  It does not combine TMA shadow requests
with ordinary L2 state and does not treat missing instrumentation order as a
zero or as an inferred order.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from jsonl_io import open_jsonl_reader


def family(name: str) -> str:
    return name.split("_", 1)[0] if "_" in name else name


def summarize(path: Path) -> dict:
    totals = {
        "captures": 0,
        "launches": 0,
        "instrumentation_order_records": 0,
        "instrumentation_order_missing_records": 0,
        "tma_source_shadow_launches": 0,
        "tma_source_shadow_order_records": 0,
    }
    families: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "launches": 0,
            "tma_source_shadow_launches": 0,
            "tma_source_shadow_order_records": 0,
        }
    )
    with open_jsonl_reader(path) as stream:
        for line in stream:
            capture = json.loads(line)
            name = str(capture.get("canonical_capture", ""))
            fam = family(name)
            totals["captures"] += 1
            for launch in capture.get("launches", []):
                totals["launches"] += 1
                families[fam]["launches"] += 1
                metrics = launch.get("trace_metrics") or {}
                coherence = metrics.get("coherence") or {}
                order_records = int(coherence.get("instrumentation_order_records", 0))
                order_missing = int(coherence.get("instrumentation_order_missing_records", 0))
                shadow_requests = int(metrics.get("tma_shadow_l2_requests", 0))
                totals["instrumentation_order_records"] += order_records
                totals["instrumentation_order_missing_records"] += order_missing
                if shadow_requests:
                    totals["tma_source_shadow_launches"] += 1
                    totals["tma_source_shadow_order_records"] += order_records
                    families[fam]["tma_source_shadow_launches"] += 1
                    families[fam]["tma_source_shadow_order_records"] += order_records
    return {
        "scope": (
            "captured TMA/ordinary ordering evidence; missing order remains "
            "unavailable and no cross-source cache projection is performed"
        ),
        "source": str(path),
        "totals": totals,
        "families": {name: dict(value) for name, value in sorted(families.items())},
        "decision": (
            "no_shared_ordering_anchor: retain the TMA source shadow as "
            "diagnostic-only and do not combine it with ordinary L2 state"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.full)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    totals = result["totals"]
    lines = [
        "# TMA/ordinary request ordering evidence",
        "",
        "Trace evidence only. Missing instrumentation order is unavailable; no",
        "TMA source shadow requests are merged into ordinary L2 state.",
        "",
        "| captures | launches | order records | order missing | TMA-source launches | TMA-source order records |",
        "|---:|---:|---:|---:|---:|---:|",
        "| {captures} | {launches} | {instrumentation_order_records} | {instrumentation_order_missing_records} | {tma_source_shadow_launches} | {tma_source_shadow_order_records} |".format(**totals),
        "",
        "| family | launches | TMA-source launches | TMA-source order records |",
        "|---|---:|---:|---:|",
    ]
    for name, value in result["families"].items():
        lines.append(
            f"| {name} | {value['launches']} | {value['tma_source_shadow_launches']} | "
            f"{value['tma_source_shadow_order_records']} |"
        )
    lines.extend(
        [
            "",
            "No shared ordering anchor is present in this replay evidence. The shadow",
            "therefore remains diagnostic/conditional rather than a cache or DRAM",
            "replay input.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
