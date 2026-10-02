#!/usr/bin/env python3
"""Report which completion-witness domains a simulator trace actually closes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "sm86_tail_witness_report_v1"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def report_tail_witness(document: Mapping[str, Any]) -> dict[str, Any]:
    launches = document.get("launches")
    if not isinstance(launches, list) or not launches:
        raise ValueError("simulator document has no launches")
    rows: list[dict[str, Any]] = []
    for index, launch_value in enumerate(launches):
        launch = _require_mapping(launch_value, f"launch {index}")
        coverage = _require_mapping(launch.get("coverage"), f"launch {index}.coverage")
        counters = _require_mapping(launch.get("counters"), f"launch {index}.counters")
        trace_metrics = _require_mapping(launch.get("trace_metrics"), f"launch {index}.trace_metrics")
        instructions = _require_mapping(
            trace_metrics.get("instructions", counters.get("instructions")),
            f"launch {index}.trace_metrics.instructions",
        )
        coherence = _require_mapping(trace_metrics.get("coherence"), f"launch {index}.trace_metrics.coherence")
        completion = _require_mapping(_require_mapping(launch.get("timing"), f"launch {index}.timing").get("completion_frontier"), f"launch {index}.completion_frontier")
        tails = _require_mapping(completion.get("unmodeled_tails", {}), f"launch {index}.unmodeled_tails")
        open_domains: list[str] = []
        facts = {
            "launch_coverage": coverage.get("kind") == "full",
            "decoded_addresses": coherence.get("unknown_sm_or_line_records") == 0,
            "active_masks": instructions.get("active_mask_missing_instructions") == 0,
            "predicate_masks": instructions.get("predicate_mask_missing_instructions") == 0,
            "cross_sm_temporal_order": coherence.get("cross_sm_lines", 0) == 0 and coherence.get("instrumentation_order_missing_records", 0) == 0,
            "line_generation": False,
            "owner_read_from": False,
            "fence_scope": False,
            "hardware_completion_token": False,
            "simulator_tail_flags_empty": completion.get("status") == "complete" and not any(tails.values()),
        }
        for domain, closed in facts.items():
            if not closed:
                open_domains.append(domain)
        rows.append({
            "launch_index": index,
            "kernel_name": launch.get("kernel_name"),
            "completion_status": completion.get("status"),
            "facts": facts,
            "open_domains": open_domains,
            "independent_tail_witness_complete": not open_domains,
            "native_dispatch_eligible": not open_domains,
        })
    return {
        "schema": SCHEMA,
        "launch_count": len(rows),
        "independent_tail_witness_complete_launches": sum(row["independent_tail_witness_complete"] for row in rows),
        "native_dispatch_eligible_launches": sum(row["native_dispatch_eligible"] for row in rows),
        "rows": rows,
        "interpretation": "missing_owner_read_from_generation_fence_scope_or_hardware_completion_stays_open",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = report_tail_witness(json.loads(args.simulator_json.read_text()))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "open_launches": report["launch_count"] - report["independent_tail_witness_complete_launches"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
