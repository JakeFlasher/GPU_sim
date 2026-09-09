#!/usr/bin/env python3
"""Bind a completed multi-kernel capture and simulator replay outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT.resolve()))
        if path.resolve().is_relative_to(REPO_ROOT.resolve())
        else str(path.resolve()),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--arch-card", type=Path, required=True)
    parser.add_argument("--pchase-card", type=Path, required=True)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    trace = args.trace.resolve()
    output = args.output.resolve()
    stats = trace / "stats.csv"
    metadata = json.loads((trace / "extra_info/traced_kernel_modules.json").read_text())
    execution = json.loads((trace / "extra_info/enhanced_execution_info.json").read_text())
    rows = [row for row in stats.read_text().splitlines()[1:] if row.strip()]
    if len(rows) != 3 or len(metadata.get("launches", [])) != 3 or len(execution.get("kernels", [])) != 3:
        raise SystemExit("multi-kernel capture does not contain exactly three launches")
    replay_outputs = {}
    for name in ("launch_cold_feedback.json", "stream_l2_no_feedback.json"):
        path = root / "simulator" / name
        doc = json.loads(path.read_text())
        if len(doc.get("launches", [])) != 3:
            raise SystemExit(f"{name}: simulator did not replay three launches")
        replay_outputs[name] = {
            "identity": identity(path),
            "launches": [
                {
                    "kernel_id": launch["kernel_id"],
                    "kernel_name": launch["kernel_name"],
                    "issue_cycles": launch["timing"]["diagnostic_last_issue_cycles"],
                    "full_terminal_cycles": launch["timing"]["terminal_modeled_event_time"],
                    "sm_active_terminal_cycles": launch["timing"]["sm_active_terminal_frontier"]["cycles"],
                }
                for launch in doc["launches"]
            ],
        }
    manifest = {
        "schema_version": 1,
        "report_id": "sm86_task_done_boundary_multikernel_manifest_v1",
        "status": "complete",
        "trace": {
            "root": str(trace.relative_to(REPO_ROOT)),
            "file_count": sum(1 for item in trace.rglob("*") if item.is_file()),
            "stats": identity(stats),
            "traced_kernel_modules": identity(trace / "extra_info/traced_kernel_modules.json"),
            "enhanced_execution_info": identity(trace / "extra_info/enhanced_execution_info.json"),
        },
        "control": identity(args.control.resolve()),
        "simulator": identity(args.simulator.resolve()),
        "arch_card": identity(args.arch_card.resolve()),
        "pchase_card": identity(args.pchase_card.resolve()),
        "feedback": identity(args.feedback.resolve()) if args.feedback else None,
        "replays": replay_outputs,
        "scientific_boundary": {
            "ncu_values_used_for_tuning": False,
            "host_copy_is_outside_trace": True,
            "launch_order_preserved": True,
            "simulator_runs_are_unscored_probe_runs": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
