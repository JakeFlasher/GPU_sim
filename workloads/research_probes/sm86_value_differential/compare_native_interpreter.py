#!/usr/bin/env python3
"""Compare the native NVBit memory/value capture with the finite interpreter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from sm86_value_trace import execute
except ModuleNotFoundError:  # pytest/import from the workload directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "nvbit_tracer"))
    from sm86_value_trace import execute


class DifferentialRefusal(RuntimeError):
    pass


def _line(document: str, prefix: str) -> dict[str, Any]:
    for raw in document.splitlines():
        if raw.startswith(prefix):
            return json.loads(raw[len(prefix) :])
    raise DifferentialRefusal(f"missing {prefix.strip()}")


def compare(sass_path: Path, capture_path: Path) -> dict[str, Any]:
    document = capture_path.read_text(encoding="utf-8")
    capture = _line(document, "CAPTURE ")
    result = _line(document, "RESULT ")
    if result.get("cc") != "8.6" or capture.get("overflow") or capture.get("memory_overflow"):
        raise DifferentialRefusal("native capture is not an admitted complete SM86 capture")
    input_base = int(result["input_address"]) & 0xFFFF_FFFF
    output_base = int(result["output_address"]) & 0xFFFF_FFFF
    constants = {
        "0:40": 0,
        "0:280": 0,
        "0:352": input_base,
        "0:360": output_base,
        "0:368": int(result["delta"]),
    }
    memory = {str((input_base + lane * 4) & 0xFFFF_FFFF): value for lane, value in enumerate(result["inputs"])}
    checked = 0
    for lane, native_lane in enumerate(capture["lanes"]):
        interpreted = execute(
            sass_path.read_bytes(),
            lane_id=lane,
            constants=constants,
            initial_memory={int(address): value for address, value in memory.items()},
        )
        native_memory = native_lane.get("memory", [])
        interp_loads = [load for record in interpreted["records"] for load in record["loads"]]
        interp_stores = [store for record in interpreted["records"] for store in record["stores"]]
        native_loads = [
            {"address": entry["address"], "value_u32": entry["value"]}
            for entry in native_memory
            if entry["kind"] == 0
        ]
        native_stores = [
            {"address": entry["address"], "value_u32": entry["value"]}
            for entry in native_memory
            if entry["kind"] == 1
        ]
        if native_loads != interp_loads or native_stores != interp_stores:
            raise DifferentialRefusal(f"lane {lane} memory differential mismatch")
        native_predicate_frame = next(frame for frame in native_lane["frames"] if frame["pc"] == 0x80)
        interpreted_predicate = next(record for record in interpreted["records"] if record["pc"] == 0x70)
        if bool(native_predicate_frame["predicates"] & 1) != bool(interpreted_predicate["predicates_after"].get("P0", False)):
            raise DifferentialRefusal(f"lane {lane} predicate differential mismatch")
        checked += 1
    return {
        "schema": "sm86_native_interpreter_differential_v1",
        "status": "ok",
        "scoring": False,
        "lanes_checked": checked,
        "checks": ["dynamic_pc_execution", "load_return_values", "store_addresses", "store_payloads", "predicate_outcomes"],
        "claims": {"native_value_semantics_match": True, "static_sass_regeneration": False, "timing": False, "physical_pipe": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass", type=Path)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.sass, args.capture)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DifferentialRefusal) as exc:
        print(json.dumps({"schema": "sm86_native_interpreter_differential_v1", "status": "refused", "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
