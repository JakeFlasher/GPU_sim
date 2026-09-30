#!/usr/bin/env python3
"""Validate one native NVBit value capture against the independent runner.

The capture and runner are intentionally separate processes.  This validator
checks device identity, complete one-warp coverage, dynamic PC sequence,
register snapshots at the load/predicate/store points, and final output bytes.
It is functional differential evidence, not a timing or pipe-identification
claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_PCS = [0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0]


class ValidationError(RuntimeError):
    pass


def _line(document: str, prefix: str) -> dict[str, Any]:
    for raw in document.splitlines():
        if raw.startswith(prefix):
            return json.loads(raw[len(prefix) :])
    raise ValidationError(f"missing {prefix.strip()}")


def validate(path: Path) -> dict[str, Any]:
    document = path.read_text(encoding="utf-8")
    capture = _line(document, "CAPTURE ")
    result = _line(document, "RESULT ")
    if result.get("cc") != "8.6":
        raise ValidationError(f"expected SM86 runner, got {result.get('cc')!r}")
    if result.get("delta") != 2_147_483_648:
        raise ValidationError("this fixture requires delta=2^31")
    if capture.get("overflow") != 0:
        raise ValidationError(f"capture overflow={capture.get('overflow')}")
    if capture.get("memory_overflow") != 0:
        raise ValidationError(f"memory capture overflow={capture.get('memory_overflow')}")
    lanes = capture.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 32:
        raise ValidationError("capture must contain exactly 32 lanes")
    inputs = result.get("inputs")
    outputs = result.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != 32 or len(outputs) != 32:
        raise ValidationError("runner must contain 32 inputs and outputs")
    checked = 0
    predicate_true = 0
    input_base = int(result["input_address"]) & 0xFFFF_FFFF
    output_base = int(result["output_address"]) & 0xFFFF_FFFF
    capture_v2 = capture.get("schema") == "native_value_capture_v2"
    if capture_v2:
        input_base = int(result["input_address"])
        output_base = int(result["output_address"])
    for lane, entry in enumerate(lanes):
        if entry.get("lane") != lane or entry.get("count") != len(EXPECTED_PCS):
            raise ValidationError(f"lane {lane} record count/identity mismatch")
        frames = entry.get("frames")
        if not isinstance(frames, list) or [frame.get("pc") for frame in frames] != EXPECTED_PCS:
            raise ValidationError(f"lane {lane} PC sequence mismatch")
        load = frames[6]
        # ISETP is PC 0x70; the predicate register becomes visible in the
        # following predicated instruction's pre-instruction snapshot.
        predicate = frames[8]
        store = frames[11]
        memory = entry.get("memory")
        if not isinstance(memory, list) or len(memory) != 2:
            raise ValidationError(f"lane {lane} memory record count mismatch")
        load_memory, store_memory = memory
        expected_input_address = input_base + lane * 4
        expected_output_address = output_base + lane * 4
        load_address = load_memory.get("address_u64", load_memory.get("address"))
        store_address = store_memory.get("address_u64", store_memory.get("address"))
        if not capture_v2:
            expected_input_address &= 0xFFFF_FFFF
            expected_output_address &= 0xFFFF_FFFF
        if (load_memory.get("pc"), load_memory.get("kind"), load_address, load_memory.get("value")) != (0x50, 0, expected_input_address, inputs[lane]):
            raise ValidationError(f"lane {lane} load address/value mismatch")
        if (store_memory.get("pc"), store_memory.get("kind"), store_address, store_memory.get("value")) != (0xB0, 1, expected_output_address, outputs[lane]):
            raise ValidationError(f"lane {lane} store address/value mismatch")
        if load["regs"][2] != inputs[lane]:
            raise ValidationError(f"lane {lane} LDG value mismatch")
        expected_predicate = int(inputs[lane] >= result["delta"])
        observed_predicate = int(predicate["predicates"] & 1 != 0)
        if observed_predicate != expected_predicate:
            raise ValidationError(f"lane {lane} predicate mismatch")
        predicate_true += observed_predicate
        if store["regs"][7] != outputs[lane]:
            raise ValidationError(f"lane {lane} store register mismatch")
        expected = (inputs[lane] - result["delta"]) & 0xFFFF_FFFF if expected_predicate else (inputs[lane] + result["delta"]) & 0xFFFF_FFFF
        if outputs[lane] != expected:
            raise ValidationError(f"lane {lane} output arithmetic mismatch")
        checked += 1
    return {
        "schema": "sm86_value_differential_validation_v1",
        "status": "ok",
        "scoring": False,
        "functional_differential": True,
        "native_device": {key: result[key] for key in ("name", "pci", "cc", "driver_api", "uuid_hex")},
        "lanes_checked": checked,
        "predicate_true_lanes": predicate_true,
        "dynamic_pcs": EXPECTED_PCS,
        "claims": {
            "value_dependent_dynamic_execution": True,
            "static_sass_regeneration": False,
            "timing_validation": False,
            "physical_pipe_reconstruction": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.capture)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        print(json.dumps({"schema": "sm86_value_differential_validation_v1", "status": "refused", "detail": str(exc)}, sort_keys=True))
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
