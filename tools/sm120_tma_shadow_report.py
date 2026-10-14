#!/usr/bin/env python3
"""Summarize address-derived TMA source shadow coverage.

This is an observation/model-coverage report. It does not consume NCU data,
change replay timing, or reinterpret the shadow as physical L2 traffic.
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
        "launches_with_source_shadow": 0,
        "requests": 0,
        "hits": 0,
        "misses": 0,
        "address_unavailable": 0,
    }
    families: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open_jsonl_reader(path) as stream:
        for line in stream:
            capture = json.loads(line)
            name = str(capture.get("canonical_capture", ""))
            fam = family(name)
            totals["captures"] += 1
            for launch in capture.get("launches", []):
                totals["launches"] += 1
                metrics = launch.get("trace_metrics") or {}
                values = {
                    "requests": int(metrics.get("tma_shadow_l2_requests", 0)),
                    "hits": int(metrics.get("tma_shadow_l2_hits", 0)),
                    "misses": int(metrics.get("tma_shadow_l2_misses", 0)),
                    "address_unavailable": int(
                        metrics.get("tma_shadow_l2_address_unavailable", 0)
                    ),
                }
                if values["requests"] or values["address_unavailable"]:
                    totals["launches_with_source_shadow"] += 1
                for key, value in values.items():
                    totals[key] += value
                    families[fam][key] += value
    return {
        "scope": "address-derived TMA source shadow; diagnostic/conditional only; NCU not used",
        "source": str(path),
        "totals": totals,
        "families": {key: dict(value) for key, value in sorted(families.items())},
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
        "# TMA source-address shadow coverage",
        "",
        "Diagnostic/conditional source-address evidence only; NCU values were not used.",
        "",
        "| captures | launches | launches with shadow | requests | hits | misses | unavailable |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| {captures} | {launches} | {launches_with_source_shadow} | {requests} | {hits} | {misses} | {address_unavailable} |".format(**totals),
        "",
        "## Family totals",
        "",
        "| family | requests | hits | misses | unavailable |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in result["families"].items():
        lines.append(
            f"| {key} | {value.get('requests', 0)} | {value.get('hits', 0)} | "
            f"{value.get('misses', 0)} | {value.get('address_unavailable', 0)} |"
        )
    lines.extend(
        [
            "",
            "The shadow uses compact source-sector identities from native TMA sidecars.",
            "It is not merged into the shared replay L2 state or timing path because",
            "the sidecar and dynamic trace do not establish cross-source event order.",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
